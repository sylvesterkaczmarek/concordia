# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for language-model sampling helpers."""

from absl.testing import absltest
from concordia.utils import sampling


class SamplingTest(absltest.TestCase):

  def test_extract_choice_response_supported_formats(self):
    self.assertEqual(sampling.extract_choice_response('a'), 'a')
    self.assertEqual(sampling.extract_choice_response('b)'), 'b')
    self.assertEqual(sampling.extract_choice_response('(c)'), 'c')
    self.assertEqual(
        sampling.extract_choice_response('The answer is (d).'), 'd'
    )

  def test_extract_choice_response_rejects_malformed_two_character_text(self):
    for sample in ('ab', 'a.', ' a', '()'):
      with self.subTest(sample=sample):
        self.assertIsNone(sampling.extract_choice_response(sample))


if __name__ == '__main__':
  absltest.main()
