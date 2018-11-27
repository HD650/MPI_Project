#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<dirent.h>
#include<math.h>
#include<stddef.h>
#include "mpi.h"

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 65

typedef char word_document_str[MAX_STRING_LENGTH];

typedef struct o {
	char word[32];
	char document[8];
	int wordCount;
	int docSize;
	int numDocs;
	int numDocsWithWord;
} obj;

typedef struct w {
	char word[32];
	int numDocsWithWord;
	int currDoc;
} u_w;

static int myCompare (const void * a, const void * b)
{
    return strcmp (a, b);
}

int main(int argc , char *argv[]){

	// variables for openmpi program
	int rank;
    int numproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);


	DIR* files;
	struct dirent* file;
	int i,j;
	int numDocs = 0, docSize, contains;
	char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH], document[MAX_DOCUMENT_NAME_LENGTH];
	
	// Will hold all TFIDF objects for all documents
	obj TFIDF[MAX_WORDS_IN_CORPUS];
	int TF_idx = 0;
	
	// Will hold all unique words in the corpus and the number of documents with that word
	u_w unique_words[MAX_WORDS_IN_CORPUS];
	int uw_idx = 0;
	
	// Will hold the final strings that will be printed out
	word_document_str strings[MAX_WORDS_IN_CORPUS];
	
	int documents_pre_node = -1;
	int extra_work_node = -1;

	// only the master node scan the directory
	if(rank==0)
	{
		//Count numDocs
		if((files = opendir("input")) == NULL){
			printf("Directory failed to open\n");
			exit(1);
		}
		while((file = readdir(files))!= NULL){
			// On linux/Unix we don't want current and parent directories
			if(!strcmp(file->d_name, "."))	 continue;
			if(!strcmp(file->d_name, "..")) continue;
			numDocs++;
		}

		// how many documents one node should work on
		documents_pre_node = numDocs / (numproc-1);
		// how many nodes should do an extra work (1 more document)
		extra_work_node = numDocs % (numproc-1);
	}
	// tell very nodes their work responsibility
	MPI_Bcast((void*)&documents_pre_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*)&extra_work_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*)&numDocs, 1, MPI_INT, 0, MPI_COMM_WORLD);	
	

	MPI_Datatype uw_type;
	int blocklens[] = {32,1,1};
	MPI_Aint indices[3];
	indices[0] = (MPI_Aint)offsetof(u_w, word);
    indices[1] = (MPI_Aint)offsetof(u_w, numDocsWithWord);
    indices[2] = (MPI_Aint)offsetof(u_w, currDoc);
    MPI_Datatype old_types[] = {MPI_CHAR,MPI_INT,MPI_INT};
    MPI_Type_struct(3,blocklens,indices,old_types,&uw_type);
    MPI_Type_commit(&uw_type);

	MPI_Comm except_root;
	if(rank == 0)
	{
		// root node will not be included in new group
		MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &except_root);
	}
	// master doesn't work on real documents
	else
	{
		// all nodes except root will be included in new group
		MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &except_root);
		// calculate the start and end point of this node
		int start, end;
		if(rank>extra_work_node)
		{
			start = 1 + extra_work_node+(rank-1)*documents_pre_node;
			end = start + documents_pre_node - 1;
		}
		else
		{
			start = 1 + (rank-1)*(documents_pre_node+1);
			end = start + documents_pre_node;
		}
		printf("[%d]starts at %d, ends at %d\n", rank, start, end);


		// Loop through each document and gather TFIDF variables for each word
		for(i=start; i<=end; i++){
			sprintf(document, "doc%d", i);
			sprintf(filename,"input/%s",document);
			FILE* fp = fopen(filename, "r");
			if(fp == NULL){
				printf("Error Opening File: %s\n", filename);
				exit(0);
			}
			
			// Get the document size
			docSize = 0;
			while((fscanf(fp,"%s",word))!= EOF)
				docSize++;
			
			// For each word in the document
			fseek(fp, 0, SEEK_SET);
			while((fscanf(fp,"%s",word))!= EOF){
				contains = 0;
				
				// If TFIDF array already contains the word@document, just increment wordCount and break
				for(j=0; j<TF_idx; j++) {
					if(!strcmp(TFIDF[j].word, word) && !strcmp(TFIDF[j].document, document)){
						contains = 1;
						TFIDF[j].wordCount++;
						break;
					}
				}
				
				//If TFIDF array does not contain it, make a new one with wordCount=1
				if(!contains) {
					strcpy(TFIDF[TF_idx].word, word);
					strcpy(TFIDF[TF_idx].document, document);
					TFIDF[TF_idx].wordCount = 1;
					TFIDF[TF_idx].docSize = docSize;
					TFIDF[TF_idx].numDocs = numDocs;
					TF_idx++;
				}
				
				contains = 0;
				// If unique_words array already contains the word, just increment numDocsWithWord
				for(j=0; j<uw_idx; j++) {
					if(!strcmp(unique_words[j].word, word)){
						contains = 1;
						if(unique_words[j].currDoc != i) {
							unique_words[j].numDocsWithWord++;
							unique_words[j].currDoc = i;
						}
						break;
					}
				}
				
				// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
				if(!contains) {
					strcpy(unique_words[uw_idx].word, word);
					unique_words[uw_idx].numDocsWithWord = 1;
					unique_words[uw_idx].currDoc = i;
					uw_idx++;
				}
			}
			fclose(fp);
		}

		// gather uw_idx to all nodes
		int* recvcounts = NULL;
		recvcounts = (int*)malloc(numproc*sizeof(int));
		MPI_Allgather((void*)&uw_idx, 1, MPI_INT, (void*)recvcounts, 1, MPI_INT, except_root);
		// according to uw_idx from all nodes, gather unique_words data to all nodes
		u_w* all_un_data = NULL;
		int* displs = NULL;
		int totlen = 0;
		displs = (int*)malloc(numproc*sizeof(int));
		displs[0] = 0;
		totlen += recvcounts[0];
		for(int i=1; i<(numproc-1); i++)
		{
			totlen += recvcounts[i];
			displs[i] = displs[i-1] + recvcounts[i-1];
		}
		all_un_data = (u_w*) malloc(totlen*sizeof(u_w));
		MPI_Allgatherv((void*)unique_words, uw_idx, uw_type, (void*)all_un_data, recvcounts, displs, uw_type, except_root);

		uw_idx = 0;
		for(i=0;i<totlen;i++)
		{
			contains = 0;
			for(j=0; j<uw_idx; j++) {
				if(!strcmp(unique_words[j].word, all_un_data[i].word)){
					contains = 1;
					unique_words[j].numDocsWithWord++;
					break;
				}
			}
				
				// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
			if(!contains) {
				strcpy(unique_words[uw_idx].word, all_un_data[i].word);
				unique_words[uw_idx].numDocsWithWord = 1;
				uw_idx++;
			}
		}
		free(recvcounts);
		free(displs);
		free(all_un_data);
	}
	
	
	// Print TF job similar to HW4/HW5 (For debugging purposes)
	printf("-------------TF Job-------------\n");
	for(j=0; j<TF_idx; j++)
		printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].wordCount, TFIDF[j].docSize);
		
	// Use unique_words array to populate TFIDF objects with: numDocsWithWord
	for(i=0; i<TF_idx; i++) {
		for(j=0; j<uw_idx; j++) {
			if(!strcmp(TFIDF[i].word, unique_words[j].word)) {
				TFIDF[i].numDocsWithWord = unique_words[j].numDocsWithWord;	
				break;
			}
		}
	}
	
	// Print IDF job similar to HW4/HW5 (For debugging purposes)
	printf("------------IDF Job-------------\n");
	for(j=0; j<TF_idx; j++)
		printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].numDocs, TFIDF[j].numDocsWithWord);
	if(rank!=0)
	{	
		// Calculates TFIDF value and puts: "document@word\tTFIDF" into strings array
		int offset = 0;
		for(j=0; j<TF_idx; j++) {
			double TF = 1.0 * TFIDF[j].wordCount / TFIDF[j].docSize;
			double IDF = log(1.0 * TFIDF[j].numDocs / TFIDF[j].numDocsWithWord);
			double TFIDF_value = TF * IDF;
			sprintf(strings[j], "%s@%s\t%.16f\n", TFIDF[j].document, TFIDF[j].word, TFIDF_value);
			// count the total string length of this node
			offset+=strlen(strings[j]);
		}
		// Sort strings and print to file
		qsort(strings, TF_idx, sizeof(char)*MAX_STRING_LENGTH, myCompare);
		int* recvcounts = NULL;
		recvcounts = (int*)malloc(numproc*sizeof(int));
		// tell every one how long this node will write to the file
		MPI_Allgather((void*)&offset, 1, MPI_INT, (void*)recvcounts, 1, MPI_INT, except_root);
		MPI_File output;
		MPI_Status status;
		MPI_File_open(except_root, "output.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output);
		// calculate the write position of this node
		offset = 0;
		for(int i=1; i<rank;i++)
			offset+=recvcounts[i-1];
		for(i=0; i<TF_idx; i++)
		{
			//MPI_File_write_at(output, offset, (void*) strings[i], strlen(strings[i]), MPI_CHAR, &status);
			MPI_File_write_at_all(output, offset, (void*) strings[i], strlen(strings[i]), MPI_CHAR, &status);
			offset+=strlen(strings[i]);
		}
		MPI_File_close(&output);
		free(recvcounts);
	}
	// all node waits here and finish at same time
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;	
}
