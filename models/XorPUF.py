import os
from helpers import get_XY, save_to_memmap
from models.ArbiterPUF import ArbiterPUF
import random

class XorPUF:
    def __init__(self, bits: int, nr: int, seed: int = 42, noise: float = 0.00):
        self.bits = bits
        self.pufs = [ArbiterPUF(bits, seed + i) for i in range(nr)]
        self.streams = nr
        self.noise = noise

    def generate_and_save_crps(self, number: int):
        responses = self.calculate_responses_with_random_challenges(number)

        X, Y = get_XY(responses)

        del responses

        directory = "crps/xor_puf/"
        os.makedirs(directory, exist_ok=True) 

        filename = f'crps/xor_puf/{self.streams}XOR_{self.bits}bit'
        save_to_memmap(X, f"{filename}_chal_{number}.memmap")
        save_to_memmap(Y, f"{filename}_resp_{number}.memmap")

    def get_response(self, challenge):
        response_bits = [puf.calculate_response(challenge).pop() for puf in self.pufs]

        response_bit = sum(response_bits) % 2

        if random.random() < self.noise:
            response_bit = 1 - response_bit

        response = challenge + [response_bit]
        return response

    def calculate_responses_with_random_challenges(self, nr):
        challenges = [[random.randint(0, 1) for _ in range(self.bits)] for _ in range(nr)]

        responses = [self.get_response(chal) for chal in challenges]

        return responses

    def majority_vote(self, nr):
        challenges = [[random.randint(0, 1) for _ in range(self.bits)] for _ in range(nr)]

        responses = []

        for chal in challenges:
            total = 0
            for _ in range(5):
                response_bit = self.get_response(chal).pop() * 2 - 1
                total += response_bit

            responses.append(chal + [1 if total > 0 else 0])

        return responses