#include "pldaplus_model.h"

namespace learning_lda {

PLDAPLUSModelForPw::PLDAPLUSModelForPw(
    int num_topics,
    const map<string, int>& local_word_index_map,
    const map<int, int>& word_pw_map,
    const map<int, int>& global_local_word_index_map,
    const int pnum, const int pwnum)
    : LDAModel(num_topics, local_word_index_map) {
  word_pw_map_ = word_pw_map;
  global_local_word_index_map_ = global_local_word_index_map;
  pnum_ = pnum;
  pwnum_ = pwnum;
}

void PLDAPLUSModelForPw::Listen() {
  int num_topics_t = num_topics();
  int pdnum = pnum_ - pwnum_;
  int count_done = 0;
  int64*  recv_buf = new int64[num_topics_t];
  int64*  send_buf = new int64[num_topics_t];
  MPI_Request req;
  MPI_Status  status;
  bool  first_flag = true;

  do {
    MPI_Recv(recv_buf, num_topics_t, MPI_LONG_LONG,
             MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int tag = status.MPI_TAG;
    int source = status.MPI_SOURCE;

    switch(tag & 3) {   // get the last two bits
      case PLDAPLUS_TAG_FETCH : {
        MPI_Wait(&req, &status);
        map<int, int>::iterator iter =
            global_local_word_index_map_.find(tag >> PLDAPLUS_TAG_LENGTH);
        if(iter != global_local_word_index_map_.end()) {
          const TopicCountDistribution&   topic_word =
              GetWordTopicDistribution(iter->second);
          topic_word.replicate(send_buf);
        }
        MPI_Isend(send_buf, num_topics_t, MPI_LONG_LONG,
                  source, tag, MPI_COMM_WORLD, &req);
        break;
      }
      case PLDAPLUS_TAG_FETCH_GLOBAL : {
        if(first_flag) {
          first_flag = false;
        } else {
          MPI_Wait(&req, &status);
        }
        const TopicCountDistribution& global_topic =
            GetGlobalTopicDistribution();
        global_topic.replicate(send_buf);
        MPI_Isend(send_buf, num_topics_t, MPI_LONG_LONG,
                  source, tag, MPI_COMM_WORLD, &req);
        break;
      }
      case PLDAPLUS_TAG_UPDATE : {
        int word_index = global_local_word_index_map_[tag >> PLDAPLUS_TAG_LENGTH];
        for(int k = 0; k < num_topics_t; ++k) {
          IncrementTopic(word_index, k, recv_buf[k]);
        }
        break;
      }
      case PLDAPLUS_TAG_DONE : {
        ++count_done;
        break;
      }
      default : {
        // tag error
      }
    }
  } while(count_done < pdnum);
  delete recv_buf;
  delete send_buf;
}

PLDAPLUSModelForPd::PLDAPLUSModelForPd(
    int num_topics,
    const map<string, int>& local_word_index_map,
    const map<int, int>& word_pw_map,
    const map<int, int>& local_global_word_index_map,
    const set<int>& word_cover,
    const int pnum, const int pwnum)
    : LDAModel(num_topics, local_word_index_map) {
  word_pw_map_ = word_pw_map;
  local_global_word_index_map_ = local_global_word_index_map;
  pnum_ = pnum;
  pwnum_ = pwnum;
	word_cover_ = word_cover;
  buf_ = new int64[num_topics];
	word_cover_topic_ = new int64[num_topics * word_cover_.size()];
	int i = 0;
	for (set<int>::iterator it = word_cover_.begin(); it != word_cover_.end(); ++it){
    word_corver_index_map_[*it] = i;
	  ++i;
	}
}

PLDAPLUSModelForPd::~PLDAPLUSModelForPd() {
  delete buf_;
}

void PLDAPLUSModelForPd::ComputeAndInit(LDACorpus* corpus) {
  for(list<LDADocument*>::const_iterator iter = corpus->begin();
      iter != corpus->end(); ++iter) {
    LDADocument* document = *iter;
    for(LDADocument::WordOccurrenceIterator iter2(document);
        !iter2.Done(); iter2.Next()) {
      IncrementTopic(iter2.Word(), iter2.Topic(), 1);
    }
  }

  for(int i = 0; i < num_words(); ++i) {
    const TopicCountDistribution&   topic_word = GetWordTopicDistribution(i);
    topic_word.replicate(buf_);
    UpdateTopicWord(i, buf_);
  }
}

void PLDAPLUSModelForPd::GetTopicWordNonblocking(int local_word_index,
                                             int64* topic_word,
                                             MPI_Request* req) {
  int global_word_index = local_global_word_index_map_[local_word_index];
  int tag = (global_word_index << PLDAPLUS_TAG_LENGTH) | PLDAPLUS_TAG_FETCH;
  int dest = word_pw_map_[global_word_index];

  MPI_Send(topic_word, 0, MPI_LONG_LONG, dest, tag, MPI_COMM_WORLD);
  MPI_Irecv(topic_word, num_topics(), MPI_LONG_LONG,
            dest, tag, MPI_COMM_WORLD, req);
}

void PLDAPLUSModelForPd::GetGlobalTopic(int64* global_topic) {
  int num_topics_t = num_topics();
  MPI_Status  status;

  MPI_Send(buf_, 0, MPI_LONG_LONG, 0, PLDAPLUS_TAG_FETCH_GLOBAL, MPI_COMM_WORLD);
  MPI_Recv(global_topic, num_topics_t, MPI_LONG_LONG,
           0, PLDAPLUS_TAG_FETCH_GLOBAL, MPI_COMM_WORLD, &status);
  for(int dest = 1; dest < pwnum_; ++dest) {
    MPI_Send(buf_, 0, MPI_LONG_LONG,
             dest, PLDAPLUS_TAG_FETCH_GLOBAL, MPI_COMM_WORLD);
    MPI_Recv(buf_, num_topics_t, MPI_LONG_LONG,
             dest, PLDAPLUS_TAG_FETCH_GLOBAL, MPI_COMM_WORLD, &status);
    for(int k = 0; k < num_topics_t; ++k) {
      global_topic[k] += buf_[k];
    }
  }
}

void PLDAPLUSModelForPd::UpdateTopicWord(int local_word_index,
                                     int64* delta_topic) {
  int global_word_index = local_global_word_index_map_[local_word_index];
  int tag = (global_word_index << PLDAPLUS_TAG_LENGTH) | PLDAPLUS_TAG_UPDATE;
  int dest = word_pw_map_[global_word_index];

  MPI_Send(delta_topic, num_topics(), MPI_LONG_LONG,
           dest, tag, MPI_COMM_WORLD);
}

void PLDAPLUSModelForPd::Done() {
  for(int i = 0; i < pwnum_; ++i) {
    MPI_Send(buf_, 0, MPI_LONG_LONG, i, PLDAPLUS_TAG_DONE, MPI_COMM_WORLD);
  }
}

void PLDAPLUSModelForPd::UpdateWordCoverTopic(int word, int64* word_topic){
	if(word_cover_.count(word) == 1){
		int index = word_corver_index_map_[word];
		memcpy(word_cover_topic_ + index, word_topic, sizeof(*word_topic) * num_topics());
	}
}
const int64* PLDAPLUSModelForPd::GetWordCoverTopic(int word){
	if(word_cover_.count(word) == 1){
		int index = word_corver_index_map_[word];
		return word_cover_topic_ + index;
	}else
		return 0;
}


}   // namespace learning_lda
