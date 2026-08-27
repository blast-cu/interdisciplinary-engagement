ROBERTA_QWEN_JUDGE_PROMPT = """You are an expert in academic citation analysis. Your task is to classify the purpose of a citation in a scientific paper.
Given the following citation context and citation text, choose the most appropriate category from the options below. Each option also has two examples to help you understand the category. Please read the category descriptions and examples carefully before making your choice.

Citation context:
{context}

Citation text:
{citation}

Candidate categories:
{options}

Respond with ONLY the category name in a JSON object, exactly as written above. Do not include any explanation.
Here is the expected output format: 
{{
    "category": "{{category_name}}"
}}
"""

JUDGE_PROMPT = """You are an expert in academic citation analysis. Your task is to classify the purpose of a citation in a scientific paper.
Given the following citation context and citation text, choose the most appropriate category from the options below. Each option also has two examples to help you understand the category. Please read the category descriptions and examples carefully before making your choice.

Citation context:
{context}

Citation text:
{citation}

Candidate categories:
{options}

Respond with ONLY the category name in a JSON object, exactly as written above. Do not include any explanation.
Here is the expected output format: 
{{
    "category": "{{category_name}}"
}}
"""

CATEGORY_DESCRIPTIONS = {
    "substantiation + basis": """
        1. The citation implicitly or explicitly provides evidence for a theoretical, empirical, or methodological claim.
	    2. The above is true AND the claim does NOT describe the state of the literature (e.g. "There exist studies on this subject", "Multiple works have adopted this approach").
	    3. The above are true AND the claim is NOT being used as the basis for the citing work's idea or method. 
	    4. The above are true AND the citation includes a specific method or idea that is being used, built upon, improved, or otherwise modified. 
        5. The above are true AND the citation is NOT used directly.
    """,
    "substantiation": """
        1. The citation implicitly or explicitly provides evidence for a theoretical, empirical, or methodological claim.
	    2. The above is true AND the claim does NOT describe the state of the literature (e.g. "There exist studies on this subject", "Multiple works have adopted this approach").
	    3. The above are true AND the claim is NOT being used as the basis for the citing work's idea or method. 
    """,
    "basis": """
        1. The citation refers to an idea being used as a basis by the citing work.
	    2. The citation referes to a method that is explicitly being built upon, improved, modified by the citing work. It is NOT used directly.
	    3. There is no explicit mention but it is very obvious from the abstract that this is integral to the citing work's idea and is thus being built upon. 
	    4. One of the above is true AND there is NO theoretical or empirical claim associated with the idea or method that motivates its use.
	""",
    "definition": """
        1. The citation helps define or explain a topic or phrase. 
	    2. The above is true AND there is no justification, validation, or proof otherwise provided. 
    """,
    "analysis": """
        1. The context offers a critique of the citation's work
        2. The citation's work is explicity compared to the citing work
        3. The citation's work is being used a baseline to compare with the citing work's approach 
        4. One of the above is true, with NO explicit mention of building upon, using, or modifying the citation's work. 
    """,
    "use": """
        1. The citation refers to a specific concrete method, not an idea, being used by the citing work. 
        2. The above is true AND there is no explicit mention of building upon, improving, or modifying the citation's method in a non-trivial way. 
    """,
    "related work": """
        1. The citation is related to the article. 
        2. The above is true AND the citation does not fall under any of the other categories. 
    """
}

CATEGORY_EXAMPLES = {
    "substantiation + basis": [
        {
            "citation_context":"Raw text within in Tweets was tokenized using a modified version of the Twokenizer (O'Connor et al., 2010). English contractions were expanded, while specific retweet tokens, username mentions, URLs, and numeric values were replaced by generic tokens. As pronoun usage tends to differ in individuals living with depression (Vedula and Parthasarathy, 2017), we removed any English pronouns from our stop word set (English Stop Words from nltk.org). Case was standardized across all tokens, with a single flag included if an entire post was made in uppercase letters.",
            "citation_text":"Vedula and Parthasarathy, 2017",
            "reasoning": 'The citation offers proof for the claim “pronoun usage tends to differ in individuals living with depression,” and this claim is used to justify their removal of English pronouns in their approach.'
        },
        {
            "citation_context":"Understanding public opinion on a contentious issue fits that description: expressions of opinion are generated from a more complex interplay of personal values and background beliefs about the world. This is a substantive real-world problem; in the US, federal agencies are required to solicit and review public comments to inform policy. Our approach is related to efforts showing that intermediate text representations are useful for interpretive work in the computational social sciences and digital humanities, where they can be aggregated to help uncover high-level themes and narratives in text collections (Bamman and Smith, 2015;Ash et al., 2022). In a similar vein, we cluster inferential decompositions of utterances that express opinion to uncover latent structure analogous to the discovery of narratives in prior work. We analyze a corpus of public comments to the U.S Food and Drug Administration (FDA) concerning the emergency authorization of COVID-19 vaccinations in children -in terms of content and goals, our application resembles the latent argument extraction of Pacheco et al. (2022), who, building on content analysis by Wawrzuta et al. (2021), clustered tweets relating to COVID-19 to facilitate effective annotation by a group of human experts. In our case, we not only discover valuable latent categories, but we are able to assign naturalistic category labels automatically in an unsupervised way.",
            "citation_text":"Ash et al., 2022",
            "reasoning":'The citation offers proof for the claim that “intermediate text representations are useful for interpretive work in the computational social sciences and digital humanities, where they can be aggregated to help uncover high-level themes and narratives in text collections.” This claim is used to justify their approach of clustering inferential decompositions of utterances that express opinion to uncover latent structure analogous to the discovery of narratives in prior work.'
        }
    ],
    "substantiation": [
        {
            "citation_context":"A battery of studies have validated the Pollyanna hypothesis that positive words occur more frequently than negative words in human expressions, using corpora ranging from Google Books to Twitter (Dodds et al., 2015;Garcia et al., 2012;Boucher and Osgood, 1969;Kloumann et al., 2012). The typical interpretation is connected with the positivity bias, which broadly denotes 1) a tendency for people to report positive views of reality, 2) a tendency to hold positive expectations, views, and memories, and 3) a tendency to favor positive information in reasoning (Carr, 2011;Augustine et al., 2011;Hoorens, 2014). However, it remains an open question whether the Pollyanna hypothesis holds in negative reviews, where the communicative goal is to express negative opinions.",
            "citation_text":"Kloumann et al., 2012",
            "reasoning":'The citation offers proof of the claim “positive words occur more frequently than negative words in human expressions.” While the fact of this evidence is mentioned explicitly in the citation context above, this explicit mention is not necessary to apply this category to a citation.'
        },
        {
            "citation_context":"Finally, we reflected on the scarcely represented labels. Among acts, Positive and Negative rhetoric and Irony appear least frequently. These labels can be broadly classified as rhetorical questions. They typically serve for self-expression than conversational engagement and, therefore, are less common than other forms of questions (Huang et al., 2017). Moreover, negative rhetorical prompts may harm the conversation quality (Zhang et al., 2018), which could also explain why listeners avoided them in empathetic dialogs. The same reasoning applies to the two infrequent intents, Pass judgement and Moralize speaker. Another surprisingly rare intent is Motivate. We believe that motivation might be difficult to express in the form of a question. Moreover, for people who did not undergo special training, expressing motivation might be more challenging than other intents as it suggests a more thorough approach to solving one's problems.",
            "citation_text":"Zhang et al., 2018",
            "reasoning":'The citation supports the claim that “negative rhetorical prompts may harm the conversation quality.”'
        }
    ],
    "basis": [
        {
            "citation_context":"This taxonomy draws on the typologies of abuse presented by Waseem et al. (2017) and Vidgen et al. (2019) as well as theoretical work in online misogyny research (Filipovic, 2007;Mantilla, 2013;Jane, 2016;Ging, 2017;Anzovino et al., 2018;Ging and Siapera, 2019;Farrell et al., 2019). It was developed by reviewing existing literature on online misogyny and then iterating over small samples of the dataset. This deductive-inductive process allowed us to ensure that conceptually distinct varieties of abuse are separated and that different types of misogyny can be unpicked. This is important given that they can have very different impacts on victims, different causes, and reflect different outlooks and interests on the part of the speaker.",
            "citation_text":"Vidgen et al., 2019",
            "reasoning":"The citation is classified as ‘Basis’ because the authors state that their taxonomy (central to their work, although this factor is not necessary) “draws on” the typology presented in the citation."
        },
        {
            "citation_context":"Strategy 1: Entity Wiki Entities in KBs often have multiple edges connecting them to other entities via relations, each representing a distinct but related fact about the entity. Inspired by the task of knowledge base completion (Bordes et al., 2013;Vashishth et al., 2019) to predict missing connections in KBs based on available KB facts, we propose the entity wiki factuality pretraining, where an LM is pretrained with the task of predicting masked entities or relations in KB facts.",
            "citation_text":"Vashishth et al., 2019",
            "reasoning":"The citation describes “the task of knowledge base completion…available KB facts,” which the authors of the citing work use as inspiration for their own approach."
        }
    ],
    "definition": [
        {
            "citation_context":"Propaganda is defined as content that intentionally influences opinion to advance its creators' goals (Bolsover and Howard, 2017). Numerous propaganda datasets have previously been created, with both document-level (Rashkin et al., 2017;Barrón-Cedeño et al., 2019) and span-level (Da San Martino et al., 2019b) technique annotations, using articles collected from multiple disinformation sites. At article-level, classifiers using combinations of multiple linguistic representations based on style and readability outperform content representation (Barrón-Cedeño et al., 2019), whereas content-based transformer models such as BERT have seen use at span-level (Da San Martino et al., 2019a). Detectors are often evaluated on single datasets, prompting concerns on generalisation (Martino et al., 2020).",
            "citation_text":"Bolsover and Howard, 2017",
            "reasoning": "The above citation offers a definition for “propaganda” that is used in the citing work."
        },
        {
            "citation_context":"Mental health stigma and gender. Mental health stigma can be defined as the negative perceptions of individuals based on their mental health status (Corrigan and Watson, 2002). This definition is implicitly composed of two pieces: assumptions about who may have mental health conditions in the first place, and assumptions about what such people are like in terms of characteristics and personality. Thus, our study at the intersection of gender bias and mental health stigma is twofold: whether models associate mental health conditions with a particular gender, and what presuppositions these models have towards different genders with mental illness.",
            "citation_text":"Corrigan and Watson, 2002",
            "reasoning": "The above citation offers a definition for “mental health stigma” that is used in the citing work."
        }
    ],
    "analysis": [
        {
            "citation_context":"If we view the AAs as a committee of three members, our architecture is reminiscent of the Query by Committee (QBC) (Seung et al., 1992), an effective approach for active learning paradigm. The essence of QBC is to select unlabeled data for labeling on which disagreement among committee members (i.e., learners pre-trained on the same labeled data) occurs. The selected data will be labeled by an oracle (e.g., domain experts) and then used to further train the learners. Likewise, in our approach, each AA votes for an item independently. However, the purpose is to detect disagreements instead of using disagreements as a measure to select items for further annotations.",
            "citation_text":"Seung et al., 1992",
            "reasoning":"The citation refers to an architecture that is similar to the one proposed in the citing work. It is clear from the citation context that the authors of the citing work are contrasting their own architecture with the one in the citation."
        },
        {
            "citation_context":"There has been accumulated a solid body of evidence that at least partially supports this causal link. The evidence comes from typological studies (Lupyan and Dale, 2010;SinnemÃ¤ki, 2009;Bentz and Winter, 2013;Bentz et al., 2015;Szmrecsanyi and Kortmann, 2009), diachronic analyses (Carroll et al., 2012), computational modeling (Reali et al., 2014) and laboratory experiments (Atkinson et al., 2019;Raviv et al., 2019;Berdicevskis and Semenuks, 2020). Nonetheless, the evidence is not entirely consistent. Studies, for instance, yield different results with respect to whether both factors (population size and proportion of L2 speakers) are at play (Sinnemaki and Di Garbo, 2018) or only one of them (Koplenig, 2019), or whether population size does play a role, but solely because it is strongly correlated with the proportion of L2 speakers. Moreover, the alleged mechanism of simplification is not fully clear.",
            "citation_text":"Reali et al., 2014",
            "reasoning":"The citation context tells us that their hypotheses are partially supported by evidence obtained from the above citation. However, this evidence is criticized in the context of the citing work’s premise."
        }
    ],
    "use": [
        {
            "citation_context":"Our second experiment predicts the amount of time that has passed in the same 250-word passage sampled above, using the conceptual framework of Underwood (2018), explored in the context of GPT-4 in Underwood (2023). Unlike the year prediction task above, which can potentially draw on encyclopedic knowledge that GPT-4 has memorized, this task requires reasoning directly about the information in the passage, as each passage has its own duration. Memorized information about a complete book could inform the prediction about all passages within it (e.g., through awareness of the general time scales present in a work, or the amount of dialogue within it), in addition to having exact knowledge of the passage itself. As above, our core question asks: do we see a difference in performance between books that GPT-4 has memorized and those that it has not? To assess this, we again identify the top and bottom decile of books by their GPT-4 name cloze accuracy. We manually annotate the amount of time passing in a 250-word passage from that book using the criteria outlined by Underwood (2018), including the examples provided to GPT-4 in Underwood (2023), blinding annotators to the identity of the book and which experimental condition it belongs to. We then pass those same passages through GPT-4 using the prompt in figure 4, adapted from Underwood (2023). We calculate accuracy as the Spearman ρ between the predicted times and true times and calculate a 95% confidence interval over that measure using the bootstrap.",
            "citation_text":"Underwood (2023)",
            "reasoning": "The above citation refers to a framework that is used directly in the citing work. There is no explicit reference to modifying this framework, or building upon it, and so it is inferred that the framework is used as is, and thus categorized as ‘Use’ and not ‘Basis’."
        },
        {
            "citation_context":"Video-to-Text Prompting. During the prompting stage, we use BLIP-2 (Li et al., 2023a), Intern-Video (Wang et al., 2022a), Whisper, ChatGPT (OpenAI, 2023), and an audio-tagging model from Schmid et al. (2022). We use the coco-pretrained BLIP-2 model with nucleus sampling. For Intern-Video, we use CLIP ViT-L/14 as the image encoder. We set the temperature to 0.3 for ChatGPT, and we use the mn40_as model for audio tagging.",
            "citation_text":"Schmid et al. (2022)",
            "reasoning": "The authors use a tool described in the citation directly in their work."        
        }
    ],
    "related work": [
        {
            "citation_context":"Although Baker et al. use human auditors to find policy keywords that minimize the false positive and false negative rates, they do not expand or optimize for economy or uncertainty keywords. Thus, we expand these keyword lists via GloVe word embeddings 9 (Pennington et al., 2014), and find the five nearest neighbors via cosine distance. 10 This is a simple keyword expansion technique. In future work, one could look to the literature on lexicon induction to improve creating lexicons that represent the semantic concepts of interest (Taboada et al., 2011;Pryzant et al., 2018;Hamilton et al., 2016;Rao and Ravichandran, 2009). Alternatively, one could also create a probabilistic classifier over pre-selected lexicons to soften the predictions, or use other uncertainty lexicons or even automatic uncertainty cue detectors.",
            "citation_text":"Hamilton et al., 2016",
            "reasoning":"As inferred from the given context, the citation describes work related to creating lexicons that represent the semantic concepts of interest, which is identifiable as related to the citing work. The ideas being referenced are potential future directions, implying a certain level of novelty; it is therefore unlikely that the citation contains a concrete example of the specified idea."
        },
        {
            "citation_context":"The idea of social roles and methods for identifying them are fairly long-standing concepts; we refer readers to Welser et al. (2007) for an in-depth review of the early history. Welser et al. (2007), analyzing Usenet forum posts, exemplifies early approaches for identifying social roles, which primarily relied on creating visualizations of authorship and reply networks then manually inspecting them to identify patterns. More recent work has leveraged more sophisticated computational models and methods such as information cascades to identify a particular role: that of social leaders and influencers (Lü et al., 2011;Tsai et al., 2014;Rosenthal, 2014). Jain et al. (2014) attempt to identify a broader set of functional roles, but their work is limited to online arguments and relies on human annotation of content. Our work attempts to provide a more general unsupervised method for identifying many functional roles in online communities, and we identify or define these roles in terms of the stylistic word choices that users make.",
            "citation_text":"Welser et al. (2007)",
            "reasoning":"The citation talks about an early approach to a task related to that primarily discussed in the citing work. There is no explicit comparison, criticism or improvement offered."
        }
    ]
}