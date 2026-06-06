
def generate_prompts(sample, lang, generate=True, likelihood=False):
    # print(sample)
    if not generate:
        return sample["multiple_choice_qa"]["question_ar"], sample["multiple_choice_qa"]["answer"], sample["question_id"], sample["source_image_file"]
    if lang=="ar":
        question = sample["multiple_choice_qa"]["question_ar"]
        answer = sample["multiple_choice_qa"]["answer"]
        choice_A = sample["multiple_choice_qa"]["choice_A_ar"]
        choice_B = sample["multiple_choice_qa"]["choice_B_ar"]
        choice_C = sample["multiple_choice_qa"]["choice_C_ar"]
        choice_D = sample["multiple_choice_qa"]["choice_D_ar"]
    elif lang=="en":
        question = sample["multiple_choice_qa"]["question_en"]
        answer = sample["multiple_choice_qa"]["answer"]
        choice_A = sample["multiple_choice_qa"]["choice_A_en"]
        choice_B = sample["multiple_choice_qa"]["choice_B_en"]
        choice_C = sample["multiple_choice_qa"]["choice_C_en"]
        choice_D = sample["multiple_choice_qa"]["choice_D_en"]
    elif lang=="dialect":
        question = sample["multiple_choice_qa"]["question_dialect"]
        answer = sample["multiple_choice_qa"]["answer"]
        choice_A = sample["multiple_choice_qa"]["choice_A_dialect"]
        choice_B = sample["multiple_choice_qa"]["choice_B_dialect"]
        choice_C = sample["multiple_choice_qa"]["choice_C_dialect"]
        choice_D = sample["multiple_choice_qa"]["choice_D_dialect"]

    if likelihood:
        prompt = f"""You are an expert in visual question answering.
You will be given an image and a question about that image.
Your task is to answer the question based on the visual content of the image.
Question: {question}
Answer: """

        return prompt, answer, sample["question_id"], sample["source_image_file"], [choice_A, choice_B, choice_C, choice_D]
    else:


        prompt = f"""You are an expert in visual question answering.
You will be given an image and a question about that image.
Your task is to answer the question based on the visual content of the image.
The question is in multiple choice format, and you need to select the correct answer from the given options.
Question: {question}
Options:
1) {choice_A}
2) {choice_B}
3) {choice_C}
4) {choice_D}
Please provide the letter of the correct answer (1, 2, 3, or 4) as your response. without any additional text.
Answer: """

        return prompt, answer, sample["question_id"], sample["source_image_file"]