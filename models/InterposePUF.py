import multiprocessing
import os
import random
from tqdm import tqdm
from helpers import get_XY, save_to_memmap
from models.XorPUF import XorPUF

class InterposePUF:
    def __init__(self, bits: int, nr1: int, nr2: int, interposition: int = None, seed: int = 44):
        self.bits = bits
        self.xor_puf1 = XorPUF(bits, nr1, seed)
        self.xor_puf2 = XorPUF(bits + 1, nr2, seed + 100)
        self.interposition = interposition

        self.nr1 = nr1
        self.nr2 = nr2

    def _save_upper_xor_crps(self, responses: list[int]):
        upper_layer_X, upper_layer_Y = get_XY(responses)

        filename = f"crps/interpose_puf/{self.nr1}_{self.nr2}XOR_upper_{self.bits}bit_"

        save_to_memmap(upper_layer_X, filename + f"chal_{len(responses)}_0noise.memmap")
        save_to_memmap(upper_layer_Y, filename + f"resp_{len(responses)}_0noise.memmap")

    def _save_lower_xor_crps(self, responses: list[int], digit: int):
        lower_layer_X, lower_layer_Y = get_XY(responses)

        filename = f"crps/interpose_puf/{self.nr1}_{self.nr2}XOR_lower_{digit}_{self.bits + 1}bit_"
        save_to_memmap(lower_layer_X, filename + f"chal_{len(responses)}_0noise.memmap")
        save_to_memmap(lower_layer_Y, filename + f"resp_{len(responses)}_0noise.memmap")

    def generate_and_save_crps(self, number: int):
        challenges = [[random.randint(0, 1) for _ in range(self.bits)] for _ in tqdm(range(number), desc="Generating challenges")]

        insert_position = self.interposition if self.interposition is not None else self.bits // 2

        directory = "crps/interpose_puf/"
        os.makedirs(directory, exist_ok=True) 

        with multiprocessing.Pool() as pool:
            upper_layer_responses = pool.map(self.xor_puf1.get_response, challenges)
        # upper_layer_responses = [self.xor_puf1.get_response(c) for c in tqdm(challenges, desc="Processing Upper XOR responses")]

        self._save_upper_xor_crps(upper_layer_responses)
        del upper_layer_responses

        challenges_1 = [c[:insert_position] + [1] + c[insert_position:] for c in challenges]
        challenges_0 = [c[:insert_position] + [0] + c[insert_position:] for c in challenges]

        del challenges

        with multiprocessing.Pool() as pool:
            lower_layer_responses_0 = pool.map(self.xor_puf2.get_response, challenges_0)
        # lower_layer_responses_0 = [self.xor_puf2.get_response(c) for c in tqdm(challenges_0, desc="Processing Lower XOR responses (0)")]
        del challenges_0
        self._save_lower_xor_crps(lower_layer_responses_0, 0)
        del lower_layer_responses_0

        with multiprocessing.Pool() as pool:
            lower_layer_responses_1 = pool.map(self.xor_puf2.get_response, challenges_1)
        # lower_layer_responses_1 = [self.xor_puf2.get_response(c) for c in tqdm(challenges_1, desc="Processing Lower XOR responses (1)")]
        del challenges_1
        self._save_lower_xor_crps(lower_layer_responses_1, 1)
        del lower_layer_responses_1

    def get_response(self, challenge: list[int]):
        response1 = self.xor_puf1.get_response(challenge).pop()

        interposed_challenge = challenge.copy()
        if self.interposition is not None:
            interposed_challenge.insert(self.interposition, response1)
        else:
            interposed_challenge.insert(len(challenge) // 2, response1)

        response2 = self.xor_puf2.get_response(interposed_challenge).pop()

        return challenge + [response2]
    
    def calculate_responses_with_random_challenges(self, nr: int):
        response = []
        challenges = []

        for _ in tqdm(range(nr), desc="Generating random challenges"):
            chal = [random.randint(0, 1) for _ in range(self.bits)]
            challenges.append(chal)

        with multiprocessing.Pool() as pool:
            response = pool.map(self.get_response, challenges)

        return response
