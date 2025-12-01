import random
import numpy as np
import math

from tqdm import tqdm

from helpers import get_XY, save_to_memmap

class ArbiterPUF:
    def __init__(self, bits: int, seed: int = 43, delays: list[float] = [], noise: float = 0.00):
        self.bits = bits
        self.random_state = np.random.RandomState(seed)
        self.delays = delays if len(delays) != 0 else [ self.random_state.uniform(-1,1) for _ in range(bits) ]
        print(f"for {seed} it is {self.delays}")
        self.noise = noise

    def generate_and_save_crps(self, number: int):
        responses = self.generate_challenges_reponses(number)

        X, Y = get_XY(responses)

        del responses

        filename = 'crps/arbiter_puf/' + str(self.bits) + "bit"

        challenges_filename = filename + "_chal_" + str(number) + ".memmap"
        responses_filename = filename + "_resp_" + str(number) + ".memmap"

        save_to_memmap(X, challenges_filename)
        save_to_memmap(Y, responses_filename)
    
    def calculate_response(self, challenge: list[int]):
        if len(challenge) != self.bits:
            raise ValueError(f"Challenge length must be {self.bits} bits.")
        
        if not all(bit in [0, 1] for bit in challenge):
            raise ValueError("Challenge must be a list of 1's and 0's.")
        
        phi = [self._calculate_phi(challenge[i:]) for i in range(len(challenge))]

        arbiter_output = np.dot(np.transpose(self.delays), phi)

        response_bit = 0 if arbiter_output < 0 else 1
        
        if random.random() < self.noise:
            response_bit = 1 - response_bit
        
        return challenge + [response_bit]
    
    def generate_challenges_reponses(self, num_responses: int):
        response = []
        
        for _ in range(num_responses):
            chal = [ random.randint(0, 1) for _ in range(self.bits) ]
            response.append(self.calculate_response(chal))
        
        return response
    
    def generate_challenges_reponses_majority(self, num_responses: int):
        response = []
        
        for _ in range(num_responses):
            chal = [ random.randint(0, 1) for _ in range(self.bits) ]

            total = 0
            for _ in range(5):
                response_bit = self.calculate_response(chal).pop() * 2 -1

                total += response_bit

            response.append(chal + [1 if total > 0 else 0])
        
        return response

    def _calculate_phi(self, challenges):
        return np.prod([math.pow(-1, c) for c in challenges])
    