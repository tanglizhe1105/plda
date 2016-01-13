#Thanks for reading README.

We improve plda on it's loglikelihood compution. We implement it as microsoft lightlda algorithm. The the motive for us to do this due to doc likelihood is decreasing by the iteration in origin code. We did many experience comfirmed it. You can test as much as you like.
###if you want to print out loglikehood, remember add "--compute_likelihood true" in command line.

There are explain of loglikelihood:
https://github.com/Microsoft/lightlda/issues/9

Let me try to make it more clear. Computing the total likelihood will need compute on all the data and model, which is prohibitively expensive. In a data-parallel distributed setting, the data is distributed across several workers and model is distributed stored across several servers.

To compute the likelihood efficiently, we decompose the total likelihood to three parts: doc-likelihood(only concern document, can compute without access of global shared model), word-likelihood(only concern about word-topic-table, can compute with only very small part of model) and normalized item in word-likelihood(only concern about the summary row).

Each worker computes the doc-likelihood on part of documents it holds, and word-likelihood of part of model it needs. Thanks to the decomposition, the computation on each worker can be also finished with multi-threads in a data-parallel fashion.

Then the total likelihood on the whole dataset should be the summation of : 1) all the doc-likelihood on each machines. 2) all the word-likelihood of each words in vocabulary. 3) the normalized likelihood.
