#include "pldaplus_sampler.h"

namespace learning_lda {

PLDAPLUSSampler::PLDAPLUSSampler(
    double alpha,
    double beta,
    PLDAPLUSModelForPd* model,
    LDAAccumulativeModel* accum_model)
    : LDASampler(alpha, beta, model, accum_model) {
  model_pd_ = model;
  alloc_buf_ = new int64[(PLDAPLUS_MAX_POOL_SIZE + 2) * (model->num_topics())];
	global_topic_ = new int64[model->num_topics()];
}

PLDAPLUSSampler::~PLDAPLUSSampler() {
  delete alloc_buf_;
	delete global_topic_;
}

void PLDAPLUSSampler::SampleNewTopicForWordInDocumentWithDistributions(
    int word_index_in_document,
    LDADocument* document_ptr,
    bool update_model,
    int64* topic_word,
    int64* global_topic,
    int64* delta_topic) {
  LDADocument::WordOccurrenceIterator iter(document_ptr);
  iter.GotoWord(word_index_in_document);
  int word_index = iter.Word();
  int num_topics_t = model_pd_->num_topics();
  int num_words_t = model_pd_->num_words();
  for(; (!iter.Done()) && (iter.Word() == word_index); iter.Next()) {
    vector<double>  new_topic_distribution;
    new_topic_distribution.clear();
    new_topic_distribution.reserve(num_topics_t);
    int old_topic = iter.Topic();

    // Generate topic distribution for word
    for(int k = 0; k < num_topics_t; ++k) {
      int current_topic_adjustment = (update_model && k == old_topic) ? -1 : 0;
      int current_word_topic_adjustment =
          current_topic_adjustment + delta_topic[k];
      double  topic_word_factor =
          topic_word[k] + current_word_topic_adjustment;
      double  global_topic_factor =
          global_topic[k] + current_word_topic_adjustment;
      double  document_topic_factor =
          document_ptr->topic_distribution()[k] + current_topic_adjustment;

      new_topic_distribution.push_back(
          (topic_word_factor + beta_) *
          (document_topic_factor + alpha_) /
          (global_topic_factor + num_words_t * beta_));
    }

    int new_topic = GetAccumulativeSample(new_topic_distribution);
    if(update_model) {
      delta_topic[new_topic] = delta_topic[new_topic] + 1;
      delta_topic[old_topic] = delta_topic[old_topic] - 1;
    }
    iter.SetTopic(new_topic);
  }
}

void PLDAPLUSSampler::DoIteration(
    PLDAPLUSCorpus* pldaplus_corpus,
    bool train_model,
    bool burn_in) {
  int num_words_t = model_pd_->num_words();
  int num_topics_t = model_pd_->num_topics();
  int64*  topic_word;
  int64*  global_topic = alloc_buf_;
  int64*  delta_topic = alloc_buf_ + num_topics_t;
  int64*  recv_buf = alloc_buf_ + (num_topics_t + num_topics_t);
  int pool_size = 0;
  int request_index;
  int*    word_index_pool = new int[PLDAPLUS_MAX_POOL_SIZE];
  MPI_Request*    request_pool = new MPI_Request[PLDAPLUS_MAX_POOL_SIZE];
  MPI_Status  status;

  // Fetch global topic distribution only once
  model_pd_->GetGlobalTopic(global_topic);

	memcpy(global_topic_, global_topic, sizeof(*global_topic) * num_topics_t);

  // Init fetching pool
  for(int i = 0; i < num_words_t && pool_size < PLDAPLUS_MAX_POOL_SIZE; ++i) {
    model_pd_->GetTopicWordNonblocking(i, recv_buf + pool_size * num_topics_t,
                                       request_pool + pool_size);
    word_index_pool[pool_size] = i;
    ++pool_size;
  }

  for(int i = pool_size; i < num_words_t; ++i) {
    // Wait for fetching any topic word distribution
    MPI_Waitany(PLDAPLUS_MAX_POOL_SIZE, request_pool, &request_index, &status);

    // Redirect topic word pointer
    topic_word = recv_buf + request_index * num_topics_t;
    memset(delta_topic, 0, sizeof(*delta_topic) * num_topics_t);

    int local_word_index = word_index_pool[request_index];
		model_pd_->UpdateWordCoverTopic(local_word_index, topic_word);
    // Sample for word local_word_index
    for(list<InvertedIndex*>::iterator iter = pldaplus_corpus->word_inverted_index[local_word_index].begin();
        iter != pldaplus_corpus->word_inverted_index[local_word_index].end(); ++iter) {
      SampleNewTopicForWordInDocumentWithDistributions(
          (*iter)->word_index_in_document,
          (*iter)->document_ptr, train_model,
          topic_word, global_topic, delta_topic);
    }

    // Update for word local_word_index
    model_pd_->UpdateTopicWord(local_word_index, delta_topic);
    for(int k = 0; k < num_topics_t; ++k) {
      global_topic[k] += delta_topic[k];
    }

    // Fetch next topic word distribution
    model_pd_->GetTopicWordNonblocking(i, topic_word, request_pool + request_index);
    word_index_pool[request_index] = i;
  }

  // Sample for the remaining words
  for(int j = 0; j < pool_size; ++j) {
    MPI_Wait(request_pool + j, &status);

    topic_word = recv_buf + j * num_topics_t;
    memset(delta_topic, 0, sizeof(*delta_topic) * num_topics_t);

    int local_word_index = word_index_pool[j];
		model_pd_->UpdateWordCoverTopic(local_word_index, topic_word);
		
    for(list<InvertedIndex*>::iterator iter = pldaplus_corpus->word_inverted_index[local_word_index].begin();
        iter != pldaplus_corpus->word_inverted_index[local_word_index].end(); ++iter) {
      SampleNewTopicForWordInDocumentWithDistributions(
          (*iter)->word_index_in_document,
          (*iter)->document_ptr, train_model, topic_word,
          global_topic, delta_topic);
    }

    model_pd_->UpdateTopicWord(local_word_index, delta_topic);
    for(int k = 0; k < num_topics_t; ++k) {
      global_topic[k] += delta_topic[k];
    }
  }

  delete word_index_pool;
  delete request_pool;
}

double PLDAPLUSSampler::ComputeOneDocLLH(LDADocument* document) const {
  const int num_topics(model_->num_topics());
  const vector<int64>& document_topic_cooccurrences(
    document->topic_distribution());
  CHECK_EQ(num_topics, document_topic_cooccurrences.size());
  
  int64 document_length = 0;
  for (int t = 0; t < num_topics; ++t) {
    document_length += document_topic_cooccurrences[t];
  }
  if(document_length == 0)
  	return 0.0;

  double one_doc_llh = LogGamma(num_topics * alpha_) 
		- num_topics * LogGamma(alpha_);
  int nonzero_num = 0;

  for (int t = 0; t < num_topics; ++t) {
    if(document_topic_cooccurrences[t] > 0){
		one_doc_llh += LogGamma(document_topic_cooccurrences[t] + alpha_);
		++nonzero_num;
    }	
  }

  one_doc_llh += (num_topics - nonzero_num) * LogGamma(alpha_);
  one_doc_llh -= LogGamma(document_length + alpha_ * num_topics);

  return one_doc_llh;
}

double PLDAPLUSSampler::ComputeOneWordLLH(int word) const{
  const int num_topics(model_->num_topics());
  const int64* word_topic_cooccurrences(
  	model_->GetWordCoverTopic(word)); //local word NO.
	CHECK_LT(0, (long)word_topic_cooccurrences);
	
	double word_llh = 0.0;
	int nonzero_num = 0;
	for (int t = 0; t < num_topics; ++t) {
		if(word_topic_cooccurrences[t] > 0){
			word_llh += LogGamma(word_topic_cooccurrences[t] + beta_);
			++nonzero_num;
    }	
  }

	if(nonzero_num == 0)
		return 0.0;
	
	word_llh += (num_topics - nonzero_num) * LogGamma(beta_);
  return word_llh;		
}

double PLDAPLUSSampler::ComputeNormalizeWordLLH() const{
  const int num_topics(model_->num_topics());
	const int num_words(model_->num_words());
	int64* global_topic_occurrences(global_topic_);
	
	double llh = num_topics * (LogGamma(num_words * beta_) -
		num_words * LogGamma(beta_));
	
	int nonzero_num = 0;
	for (int t = 0; t < num_topics; ++t) {
		if(global_topic_occurrences[t] > 0){
			llh -= LogGamma(global_topic_occurrences[t] + num_words * beta_);
			++nonzero_num;
		}
  }
	
	if(nonzero_num == 0)
		return 0.0;
	
  return llh;
}


} // namespace learning_lda
