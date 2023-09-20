# Copyright 2020 DeepMind Technologies Limited.
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
"""Bot policy implementations."""

import abc
from typing import Generic, Tuple, TypeVar


State = TypeVar('State')


class Policy(Generic[State], metaclass=abc.ABCMeta):
  """Abstract base class for a policy.

  Must not possess any mutable state not in `initial_state`.
  """
  def __init__(self, policy_id:str , policy_name:str):
      # You can implement any init operations here or in setup()
      self.policy_id = policy_id # TODO - Should this be multiple or indiviaul, current is not individual
      self.policy_name = policy_name # TODO - Should this be multiple or indiviaul, current is not individual


  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs
    self.close()