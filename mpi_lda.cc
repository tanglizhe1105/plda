#include "mpi.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <string>

#include "common.h"
#include "document.h"
#include "model.h"
#include "accumulative_model.h"
#include "sampler.h"
#include "cmd_flags.h"
#include "pldaplus_model.h"
#include "pldaplus_sampler.h"

#define PLDAPLUS_MAX_DOCUMENTS 10000000

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::stringstream;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::sort;
using std::string;

namespace learning_lda {

int DistributelyLoadAndInitTrainingCorpus(
    const string& corpus_file,
    int num_topics,
    int myid, int pnum, int pwnum,
    LDACorpus* corpus,
    set<string>* words,
    set<string>* localwords) {
  ifstream    fin(corpus_file.c_str());
  string  line;
  int index = 0;
  int pdnum = pnum - pwnum;

  if (myid >= pwnum) {
    // Processor pd
    myid -= pwnum;
    corpus->clear();
    while(index < PLDAPLUS_MAX_DOCUMENTS && getline(fin, line)) {
      if (line.size() > 0 &&
          line[0] != '\r' &&
          line[0] != '\n' &&
          line[0] != '#') {
          // Skip empty line
        istringstream   ss(line);	
        if (index % pdnum == myid) {
		          // The document that i need to store. Randomly assign topic for word.
          DocumentWordTopicsPB document;
          string  word_s;
          int count;
          while(ss >> word_s >> count) {
            vector<int32>   topics;
            for (int i = 0; i < count; ++i) {
              topics.push_back(RandInt(num_topics));
            }
            document.add_wordtopics(word_s, -1, topics);
            localwords->insert(word_s);
            words->insert(word_s);
          }
          if (document.words_size() > 0) {
			  int documentId = index;
            corpus->push_back(new LDADocument(document, num_topics, documentId));
          }
        } else {
          // The document that i don't need to store. Only read words.
          string  word_s;
          int count;
          while(ss >>word_s >> count) {
            words->insert(word_s);
          }
        }
        ++index;
      }
    }
    return corpus->size();
  } else {
    // Processor pw
    while(index < PLDAPLUS_MAX_DOCUMENTS && getline(fin, line)) {
      if (line.size() > 0 &&
          line[0] != '\r' &&
          line[0] != '\n' &&
          line[0] != '#') {
          // Skip empty line
        istringstream   ss(line);
        string  word_s;
        int count;
        while(ss >> word_s >> count) {
          words->insert(word_s);
        }
        ++index;
      }
    }
    return 1;
  }
}

void InitWordPlacement(map<int,int>& word_pw_map,
                       int num_words, int pwnum, int myid) {
  int pw_index;
  for (int i = 0; i < num_words; ++i) {
    if (myid == 0) {
      pw_index = RandInt(pwnum);
    }
    MPI_Bcast(&pw_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    word_pw_map[i] = pw_index;
  }
}

void InitWordInvertedIndex(PLDAPLUSCorpus* pldaplus_corpus) {
  list<InvertedIndex*>*   word_inverted_index = new list<InvertedIndex*>[pldaplus_corpus->num_words];
  for (list<LDADocument*>::const_iterator iter = pldaplus_corpus->corpus->begin();
      iter != pldaplus_corpus->corpus->end(); ++iter) {
    const vector<int>&  words = (*iter)->topics().words_;
    for (int i = 0; i < words.size(); ++i) {
      word_inverted_index[words[i]].push_back(new InvertedIndex(*iter, i));
    }
  }
  pldaplus_corpus->word_inverted_index = word_inverted_index;
}

void FreeCorpus(PLDAPLUSCorpus* pldaplus_corpus) {
  for (list<LDADocument*>::iterator iter = pldaplus_corpus->corpus->begin();
      iter != pldaplus_corpus->corpus->end(); ++iter) {
    if (*iter != NULL) {
      delete *iter;
      *iter = NULL;
    }
  }
  for (int i = 0; i < pldaplus_corpus->num_words; ++i) {
    for (list<InvertedIndex*>::iterator iter2 = pldaplus_corpus->word_inverted_index[i].begin();
        iter2 != pldaplus_corpus->word_inverted_index[i].end(); ++iter2) {
      if (*iter2 != NULL) {
        delete *iter2;
        *iter2 = NULL;
      }
    }
    pldaplus_corpus->word_inverted_index[i].clear();
  }
}

} // namespace learning_lda

//tlz
char* getCurTime(char ts[]){
        time_t timep;
        time (&timep);
        struct tm* tmp = localtime(&timep);
        sprintf(ts, "%02d-%02d-%02d %02d:%02d:%02d", tmp->tm_year+1900, tmp->tm_mon+1, 
			tmp->tm_mday, tmp->tm_hour, tmp->tm_min, tmp->tm_sec);
        return ts;
}

int main(int argc, char** argv) {
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::LDASampler;
  using learning_lda::DistributelyLoadAndInitTrainingCorpus;
  using learning_lda::InitWordPlacement;
  using learning_lda::LDACmdLineFlags;
  using learning_lda::PLDAPLUSCorpus;
  using learning_lda::PLDAPLUSModelForPd;
  using learning_lda::PLDAPLUSModelForPw;
  using learning_lda::PLDAPLUSSampler;

  int myid, pnum, pwnum;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum);

  // Parse arguments
  LDACmdLineFlags flags;
  flags.ParseCmdFlags(argc, argv);
  if (!flags.CheckParallelTrainingValidity()) {
    return -1;
  }
  if (flags.num_pw_ <= 0 || flags.num_pw_ >= pnum) {
    printf("Invalid num of pw\n");
    return -1;
  }
  pwnum = flags.num_pw_;

  int num_topics = flags.num_topics_;
  int num_words;

  // Define a communication group for pd's
  MPI_Group   MPI_GROUP_WORLD;
  MPI_Group   MPI_GROUP_PD;
  MPI_Comm    MPI_COMM_PD;
  int *pw_ranks = new int[flags.num_pw_];
  for (int i = 0; i < flags.num_pw_; ++i) {
    pw_ranks[i] = i;
  }
  MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
  MPI_Group_excl(MPI_GROUP_WORLD, flags.num_pw_, pw_ranks, &MPI_GROUP_PD);
  MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_PD, &MPI_COMM_PD);
  delete pw_ranks;

  char tm[50];  //tlz
  
  srand(time(NULL));
  set<string> allwords;
  set<string> localwords;
  LDACorpus   *lda_corpus = NULL;
  if (myid >= pwnum) {
    lda_corpus = new LDACorpus;
  }
  CHECK_GT(DistributelyLoadAndInitTrainingCorpus(flags.training_data_file_,
                                                 num_topics,
                                                 myid, pnum, flags.num_pw_,
                                                 lda_corpus,
                                                 &allwords, &localwords), 0);
  if (myid >= pwnum) {
    printf("[%s] : Rank %d : num of documents = %ld\n", getCurTime(tm), myid, lda_corpus->size());  //tlz
  }

  // Sort vocabulary words and give each word an int index
  vector<string>  sorted_words;
  map<string, int>    word_index_map;
  map<int, int>   word_pw_map;
  for (set<string>::const_iterator iter = allwords.begin();
      iter != allwords.end(); ++iter) {
    sorted_words.push_back(*iter);
  }
  sort(sorted_words.begin(), sorted_words.end());
  num_words = sorted_words.size();
  for (int i = 0; i < num_words; ++i) {
    word_index_map[sorted_words[i]] = i;
  }

  // Give each word a pw processor's index
  InitWordPlacement(word_pw_map, num_words, pwnum, myid);

  if (myid >= pwnum) {
    // Processor pd
    map<string, int>    local_word_index_map;
    map<int, int>   local_global_word_index_map;
    int num_local_words = 0;
    for (set<string>::const_iterator iter = localwords.begin();
        iter != localwords.end(); ++iter) {
      local_word_index_map[*iter] = num_local_words;
      local_global_word_index_map[num_local_words] = word_index_map[*iter];
      ++num_local_words;
    }

    for (LDACorpus::iterator iter = lda_corpus->begin();
        iter != lda_corpus->end(); ++iter) {
      (*iter)->ResetWordIndex(local_word_index_map);
    }
    PLDAPLUSCorpus  pldaplus_corpus;
    pldaplus_corpus.corpus = lda_corpus;
    pldaplus_corpus.num_words = num_local_words;
    InitWordInvertedIndex(&pldaplus_corpus);

    PLDAPLUSModelForPd  model_pd(num_topics,
                             local_word_index_map,
                             word_pw_map,
                             local_global_word_index_map,
                             pnum, pwnum);
    model_pd.ComputeAndInit(pldaplus_corpus.corpus);
    printf("[%s] : Rank %d : Training data loaded.\n", getCurTime(tm), myid);  //tlz
    MPI_Barrier(MPI_COMM_PD);

    // Do iteration
    PLDAPLUSSampler sampler(flags.alpha_, flags.beta_, &model_pd, NULL);
    for (int i = 0; i < flags.total_iterations_; ++i) {
      //tlz print out doc loglikelihood
	  if(flags.compute_likelihood_ == "true"){
	  	if(i != 0 && i % 5 == 0){
			double loglikelihood_local = 0;
      		double loglikelihood_global = 0;
			for (LDACorpus::iterator iter = lda_corpus->begin(); iter != lda_corpus->end(); ++iter) {
				loglikelihood_local += sampler.LogLikelihood(*iter);
			}
			MPI_Allreduce(&loglikelihood_local, &loglikelihood_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_PD);
			if (myid == pwnum)
		  		printf("[%s] : Rank %d : iteration %d : loglikelihood %e\n", getCurTime(tm), myid, i, loglikelihood_global);
	  	}
	  }
	  printf("[%s] : Rank %d : iteration %d\n", getCurTime(tm), myid, i);
	  //tlz
      sampler.DoIteration(&pldaplus_corpus, true, false);
    }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::stringstream sout;
    sout<<"output"<<myid;
    std::ofstream out(sout.str().c_str());
    outputPLDAPLUSCorpus(&pldaplus_corpus, out);
    // Inform pw of completion
    model_pd.Done();

    FreeCorpus(&pldaplus_corpus);
  } else {
    // Processor pw
    map<string, int>    local_word_index_map;
    map<int, int>   global_local_word_index_map;
    int num_local_words = 0;
    for (int i = 0; i < num_words; ++i) {
      if (myid == word_pw_map[i]) {
        global_local_word_index_map[i] = num_local_words;
        local_word_index_map[sorted_words[i]] = num_local_words;
        ++num_local_words;
      }
    }
    printf("[%s] : Rank %d : num of words = %d\n", getCurTime(tm), myid, num_local_words);  //tlz

    PLDAPLUSModelForPw  model_pw(num_topics,
                             local_word_index_map,
                             word_pw_map,
                             global_local_word_index_map,
                             pnum, pwnum);
    model_pw.Listen();

    // Save results to files
    stringstream    ss_filename;
    ss_filename << flags.model_file_ << "_" << myid;
    ofstream    fout(ss_filename.str().c_str());
    model_pw.AppendAsString(fout);
  }

  printf("[%s] : Rank %d : Done\n", getCurTime(tm), myid);  //tlz
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Group_free(&MPI_GROUP_PD);
  MPI_Group_free(&MPI_GROUP_WORLD);
  MPI_Finalize();
  return 0;
} // end main
