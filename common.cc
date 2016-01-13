// Copyright 2008 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>

#include "common.h"


char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

namespace learning_lda {

bool IsValidProbDistribution(const TopicProbDistribution& dist) {
  const double kUnificationError = 0.00001;
  double sum_distribution = 0;
  for (int k = 0; k < dist.size(); ++k) {
    sum_distribution += dist[k];
  }
  return (sum_distribution - 1) * (sum_distribution - 1)
      <= kUnificationError;
}

int GetAccumulativeSample(const vector<double>& distribution) {
  double distribution_sum = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    distribution_sum += distribution[i];
  }

  double choice = RandDouble() * distribution_sum;
  double sum_so_far = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    sum_so_far += distribution[i];
    if (sum_so_far >= choice) {
      return i;
    }
  }

  LOG(FATAL) << "Failed to choose element from distribution of size "
             << distribution.size() << " and sum " << distribution_sum;

  return -1;
}

std::ostream& operator << (std::ostream& out, vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i] << " ";
  }
  return out;
}

const double cof[6] = 
{ 
		76.18009172947146, -86.50532032941677,
		24.01409824083091, -1.231739572450155,
		0.1208650973866179e-2, -0.5395239384953e-5
};

double LogGamma(double xx)
{
		int j;
		double x, y, tmp1, ser;
		y = xx;
		x = xx;
		tmp1 = x + 5.5;
		tmp1 -= (x + 0.5)*log(tmp1);
		ser = 1.000000000190015;
		for (j = 0; j < 6; j++) ser += cof[j] / ++y;
		return -tmp1 + log(2.5066282746310005*ser / x);
}

}  // namespace learning_lda
