import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """1""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Hallucination Mitigation in Large Language Models with Knowledge Graphs""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to design, develop, and evaluate a lightweight Retrieval-Augmented Generation (RAG) system 
            that integrates small-scale Knowledge Graphs (KGs) to mitigate hallucinations in open-source LLMs (e.g., TinyLLaMA or 
            DistilBERT). The project focuses on: Customizing a lightweight LLM using RAG to incorporate KG-based factual grounding 
            for question answering and text generation tasks. Evaluating hallucination reduction by comparing LLM outputs with 
            and without KG context, using metrics like factual accuracy and BLEU score. Optimizing the pipeline for local execution 
            on a MacBook M3 Pro, leveraging Apple Silicon’s MLX framework for efficient inference. Developing a small-scale, 
            domain-specific KG (e.g., from Wikipedia or academic papers) to support the RAG system."
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            "The dataset will consist of a small corpus of open-source texts, such as a subset of Wikipedia articles or academic 
            papers (<100MB), processed to create a KG with <5,000 nodes using tools like spaCy or REBEL. The FELM dataset 
            (for sentence-level hallucination detection) or a custom dataset of 100–200 question-answer pairs will be used 
            for evaluation. All data will be stored locally in compressed formats (e.g., JSON or Parquet) to fit within the 
            M3 Pro’s 512GB–1TB SSD."
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Hallucinations in LLMs, where models generate factually incorrect outputs, limit their reliability in 
            knowledge-intensive tasks. KGs provide structured, verifiable knowledge to ground LLM outputs, reducing errors. 
            This project explores how lightweight KGs can enhance LLM performance on resource-constrained devices like the 
            MacBook M3 Pro, offering a practical approach to improve factual accuracy in applications like personal knowledge 
            management or academic research. By focusing on small-scale, locally executable systems, the project addresses 
            computational limitations while contributing to the growing field of KG-augmented NLP.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will proceed in several phases:**Requirement Analysis**: Define key requirements for hallucination 
            mitigation, including KG size, LLM selection, and evaluation metrics, in collaboration with lead researchers. 
            **Development**: Build a small-scale KG using spaCy or REBEL for entity/relation extraction from a text corpus. 
            Implement a RAG pipeline with MLX and a quantized LLM (e.g., 4-bit TinyLLaMA), using FAISS for embedding-based retrieval. 
            **Validation**: Evaluate hallucination reduction on a test set (e.g., FELM or custom Q&A dataset), measuring factual 
            accuracy, BLEU score, and response time. Compare outputs with and without KG context.\n**Optimization**: Optimize 
            the pipeline for the M3 Pro by reducing KG size, quantizing models, and caching embeddings. Use Instruments to 
            monitor resource usage.**Documentation**: Document findings and prepare a report on the effectiveness of KG-based 
            hallucination mitigation.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project: 
            **Weeks 1-2**: Familiarize with hallucination mitigation literature, KG construction tools, and MLX framework. 
            **Weeks 3-6**: Develop the KG and RAG pipeline, including entity extraction and embedding generation. 
            **Weeks 7-10**: Conduct iterative testing and optimization of the RAG system, focusing on hallucination reduction. 
            **Weeks 11-14**: Evaluate performance on test datasets, comparing KG-augmented and baseline LLM outputs. 
            **Weeks 15-16**: Finalize documentation, reporting, and presentation of results.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for 1-2 students, given the focused scope and need for expertise in NLP, KGs, and lightweight model optimization."
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges include: 
            **Limited Dataset Size**: Small-scale datasets may limit KG coverage, requiring careful selection of domain-specific 
            texts. **Computational Constraints**: The M3 Pro’s 18–36GB memory limits model and KG size, necessitating quantization 
            and optimization. **Evaluation Complexity**: Measuring hallucination reduction requires robust metrics and manual 
            validation, which may be time-intensive. **Tool Compatibility**: Ensuring compatibility of open-source tools 
            (e.g., MLX, spaCy, FAISS) with Apple Silicon may require troubleshooting.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Refferences":
            """
            Lavrinovics et al., “Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspective” (2024), arXiv: https://arxiv.org/abs/2411.13409.
            Shi et al., “Hallucination Mitigation in Natural Language Generation from Large-Scale Open-Domain Knowledge Graphs” (2023), EMNLP: https://aclanthology.org/2023.emnlp-main.773, code at https://github.com/idirlab/graphnarrator.
            “From Hallucinations to Facts: Enhancing Language Models with Curated Knowledge Graphs” (2024), arXiv: https://arxiv.org/abs/2403.09909.
            “Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey” (2023), arXiv: https://arxiv.org/abs/2311.07914.
            “Training Language Models on the Knowledge Graph: Insights on Hallucinations and Their Detectability” (2024), OpenReview: https://openreview.net/forum?id=mL9cJ5kiA4.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Timur Abdygulov",
        "Proposed by email": "timur.abdygulov@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/GNN_research",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
