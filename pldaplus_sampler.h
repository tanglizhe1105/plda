#ifndef _PLDAPLUS_SAMPLER_H_
#define _PLDAPLUS_SAMPLER_H_

#include "pldaplus_model.h"
#include "sampler.h"

namespace learning_lda {

// The PLDAPLUSSampler is simply a modified version of LDASampler.
// Since the pd processer does not store the distribution data in
// the local memory, and sampling is based on word order instead
// of document order, we rewrote the DoIteration and
// SampleNewTopicForDocument methods.
class PLDAPLUSSampler : public LDASampler {
 public:
  PLDAPLUSSampler(double alpha,
              double beta,
              PLDAPLUSModelForPd* model,
              LDAAccumulativeModel* accum_model);
  ~PLDAPLUSSampler();

  // Each time do sampling for all occurrences of a word in the local
  // corpus, and distributions are fed by arguments.
  void SampleNewTopicForWordInDocumentWithDistributions(
      int word_index_in_document,
      LDADocument* document_ptr,
      bool update_model,
      int64* topic_word,
      int64* global_topic,
      int64* delta_topic);

  // Fetch distributions from pw processors before sampling
  // and update distributions on pw processors after sampling.
  void DoIteration(PLDAPLUSCorpus* pldaplus_corpus,
                   bool training_model,
                   bool burn_in);
 private:
  int64*  alloc_buf_;
  PLDAPLUSModelForPd* model_pd_;
};

} // namespace learning_lda

#endif // _PLDAPLUS_SAMPLER_H_
