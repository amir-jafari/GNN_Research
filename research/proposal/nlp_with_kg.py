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
            """Explainable NLP with Knowledge Graph Visualization""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to design, develop, and evaluate a lightweight system that enhances the explainability of 
            NLP tasks by visualizing knowledge graph (KG) reasoning paths for large language model (LLM) outputs. The project 
            focuses on: Constructing a small-scale KG (<5,000 nodes) from a domain-specific text corpus to support explainable 
            question answering and semantic search. Developing a visualization pipeline using lightweight tools (e.g., 
            Matplotlib, Plotly) to render KG subgraphs for query-specific reasoning paths. Evaluating the interpretability 
            of LLM outputs (e.g., using TinyLLaMA or MiniLM) with metrics like user satisfaction and explanation clarity. 
            Optimizing the system for local execution on a MacBook M3 Pro, leveraging the MLX framework for efficient inference 
            and Apple Silicon’s GPU for visualization."
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            "The dataset will include a small corpus of open-source texts, such as a subset of Wikipedia articles, academic papers, 
            or personal notes (<100MB), processed to create a KG using spaCy or REBEL. The HotpotQA dataset, with multi-hop 
            question-answer pairs, will be used for evaluation, subsetted to <500MB to fit the M3 Pro’s storage (512GB–1TB SSD). 
            All data will be stored locally in JSON or Parquet formats for efficiency."
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Explainability is critical for building trust in NLP systems, particularly for applications in education, research, 
            and personal knowledge management. Visualizing KG reasoning paths provides transparent insights into how LLMs derive 
            answers, making outputs more interpretable. This project explores lightweight KG visualization techniques to enhance 
            NLP explainability on resource-constrained devices like the MacBook M3 Pro, contributing to the growing demand for 
            interpretable AI systems in academic and practical settings.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will proceed in several phases: **Requirement Analysis**: Collaborate with lead researchers to define 
            requirements for explainable NLP, including KG size, visualization formats, and interpretability metrics. 
            **Development**: Build a small-scale KG using spaCy or REBEL for entity/relation extraction. Implement a pipeline 
            to integrate a lightweight LLM (e.g., MiniLM) with KG traversal, using MLX for inference. **Visualization**: Develop 
            a visualization module with Matplotlib or Plotly to render query-specific KG subgraphs, highlighting entities and 
            relations. **Validation**: Evaluate interpretability on HotpotQA or a custom Q&A dataset (100–200 questions), 
            using metrics like user satisfaction (via survey) and explanation clarity (via annotation). **Optimization**: 
            Optimize the pipeline for the M3 Pro by reducing KG size, quantizing models (e.g., 4-bit), and caching embeddings. 
            Use Instruments to monitor resource usage. **Documentation**: Document findings and prepare a report on the 
            effectiveness of KG visualization for explainable NLP.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            his is a rough timeline for the project: 
            **Weeks 1-2**: Familiarize with explainable NLP literature, KG visualization tools, and MLX framework. 
            **Weeks 3-6**: Develop the KG and visualization pipeline, including entity extraction and subgraph rendering. 
            **Weeks 7-10**: Conduct iterative testing of the visualization system, refining based on feedback from sample queries. 
            **Weeks 11-14**: Evaluate interpretability on test datasets, comparing visualized vs. non-visualized LLM outputs. 
            **Weeks 15-16**: Finalize documentation, reporting, and presentation of results.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for 1-2 students, given the focused scope and need for expertise in NLP, KGs, and visualization techniques.            
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges include: 
            **Limited Dataset Coverage**: Small-scale datasets may restrict KG comprehensiveness, requiring careful text selection. 
            **Visualization Complexity**: Rendering clear and informative KG subgraphs may require iterative design to balance detail 
            and simplicity. **Computational Constraints**: The M3 Pro’s 18–36GB memory limits model and KG size, necessitating 
            quantization and optimization. **Subjective Metrics**: Evaluating interpretability (e.g., user satisfaction) may be 
            subjective, requiring robust survey design or annotation protocols.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Refferences":
            """
            “Explainable Hallucination Mitigation in Large Language Models: A Survey” (2025), Preprints.org: https://www.preprints.org/manuscript/202505.0119/v1.
            “A Preliminary Roadmap for LLMs as Assistants in Exploring, Analyzing, and Visualizing Knowledge Graphs” (2024), arXiv: https://arxiv.org/abs/2404.12896.
            “FOKE: A Personalized and Explainable Education Framework Integrating Foundation Models, Knowledge Graphs, and Prompt Engineering” (2024), arXiv: https://arxiv.org/abs/2405.16367.
            “Supporting Student Decisions on Learning Recommendations: An LLM-Based Chatbot with Knowledge Graph Contextualization for Conversational Explainability and Mentoring” (2024), arXiv: https://arxiv.org/abs/2401.08517.
            “GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework” (2024), Amazon Science: https://www.amazon.science/publications/grapheval-a-knowledge-graph-based-llm-hallucination-evaluation-framework.
            “Knowledge Graph Treatments for Hallucinating Large Language Models” (2024), ERCIM News: https://ercim-news.ercim.eu/en136/r-i/knowledge-graph-treatments-for-hallucinating-large-language-models.
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
