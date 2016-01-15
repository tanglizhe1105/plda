#ifndef _PLDAPLUS_MODEL_H_
#define _PLDAPLUS_MODEL_H_

#include "mpi.h"

#include "document.h"
#include "model.h"


#define PLDAPLUS_MAX_POOL_SIZE 100

// The PLDAPLUS_TAG_xxxx facilities, which is used for packing and
// parsing different types of MPI tags.
#define PLDAPLUS_TAG_LENGTH 2
#define PLDAPLUS_TAG_FETCH 0
#define PLDAPLUS_TAG_FETCH_GLOBAL 1
#define PLDAPLUS_TAG_UPDATE 2
#define PLDAPLUS_TAG_DONE 3

namespace learning_lda {

// The PLDAPLUSModelForPw class stores distributed words with their
// topic distributions and a vector of 'global' topic occurrence
// counts of all the words on itself.
//
// This class process, fetch (either for word-topic or global
// distribution), and update queries from pd processors
class PLDAPLUSModelForPw : public LDAModel {
 public:
  PLDAPLUSModelForPw(int num_topics,
                 const map<string, int>& local_word_index_map,
                 const map<int, int>& word_pw_map,
                 const map<int, int>& global_local_word_index_map,
                 const int pnum, const int pwnum);
  void Listen();
	void ComputeLocalWordLlh(); //tlz
 private:
  map<int, int> word_pw_map_;
  map<int, int> global_local_word_index_map_;
  int pnum_, pwnum_;
};

// The PLDAPLUSModelForPd class does not store any distributions locally.
//
// This class provides methods to initialize word-topic distributions
// on pw processors, fetch word-topic or global-topic distributions
// from pw processors, and update word-topic distributions.
class PLDAPLUSModelForPd : public LDAModel {
 public:
  PLDAPLUSModelForPd(int num_topics,
                 const map<string, int>& local_word_index_map,
                 const map<int, int>& word_pw_map,
                 const map<int, int>& local_global_word_index_map,
                 const set<int>& word_cover,
                 const int pnum, const int pwnum);
  ~PLDAPLUSModelForPd();

  // Compute topic word co-occurrence and initialize processor pw.
  void ComputeAndInit(LDACorpus* corpus);

  // Get topic word distribution. Don't wait for communication's completion.
  void GetTopicWordNonblocking(int local_word_index,
                               int64* topic_word, MPI_Request* req);

  // Get global topic distribution.
  void GetGlobalTopic(int64* global_topic);

  // Update topic word distribution.
  void UpdateTopicWord(int local_word_index, int64* delta_topic);

  // Inform processor pw of completion
  void Done();
	
	// Word cover means the vocabulary of corpus that divided by document model process.
	// Each document model process possess part vocabulary and all document model process
	// cover total vocabulary.
	// Update local word-topic with value from word-topic modle.
	void UpdateWordCoverTopic(int word, int64* word_topic);
	// Get local word-topic which value from word-topic modle.
	const int64* GetWordCoverTopic(int word);
	
 private:
  int64*  buf_;
	int64*	word_cover_topic_;
	set<int> word_cover_;
	map<int, int> word_corver_index_map_;
  map<int, int> word_pw_map_;
  map<int, int> local_global_word_index_map_;
  int pnum_, pwnum_;
};

}   // namespace learning_lda

#endif // _PLDAPLUS_MODEL_H_
