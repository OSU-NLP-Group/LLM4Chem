# LlaSMol
This is the official code repository for the paper *LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset*.

- Paper: https://arxiv.org/abs/2402.09391
- Page: https://osu-nlp-group.github.io/LLM4Chem
- Code: https://github.com/OSU-NLP-Group/LLM4Chem
- Models: https://huggingface.co/osunlp/LLM4Chem

## Tasks and Examples

Our models are trained on 4 types of 14 essential tasks, which, along with examples, are listed below.

### Name Conversion
**Name Conversion - IUPAC to Molecular Formula**
```
Query: What is the molecular formula of the compound with this IUPAC name <IUPAC> 2,5-diphenyl-1,3-oxazole </IUPAC> ?

Response: <MOLFORMULA> C15H11NO </MOLFORMULA>
```

**Name Conversion - IUPAC to SMILES**
```
Query: Could you provide the SMILES for <IUPAC> 4-ethyl-4-methyloxolan-2-one </IUPAC> ?

Response: Of course. It's <SMILES> CCC1(C)COC(=O)C1 </SMILES> .
```

**Name Conversion - SMILES to Molecular Formula**
```
Query: Given the SMILES representation <SMILES> S=P1(N(CCCl)CCCl)NCCCO1 </SMILES>, what would be its molecular formula?

Response: It is <MOLFORMULA> C7H15Cl2N2OPS </MOLFORMULA> .
```

**Name Conversion - SMILES to IUPAC**
```
Query: Translate the given SMILES formula of a molecule <SMILES> CCC(C)C1CNCCCNC1 </SMILES> into its IUPAC name.

Response: <IUPAC> 3-butan-2-yl-1,5-diazocane </IUPAC>
```

### Property Prediction
**Property Prediction - ESOL**
```
Query: How soluble is <SMILES> CC(C)Cl </SMILES> ?

Response: Its log solubility is <NUMBER> -1.41 </NUMBER> mol/L.
```

**Property Prediction - LIPO**
```
Query: Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <SMILES> NC(=O)C1=CC=CC=C1O </SMILES> .

Response: <NUMBER> 1.090 </NUMBER>
```

**Property Prediction - BBBP**
```
Query: Is blood-brain barrier permeability (BBBP) a property of <SMILES> CCNC(=O)/C=C/C1=CC=CC(Br)=C1 </SMILES> ?

Response: <BOOLEAN> Yes </BOOLEAN>
```

**Property Prediction - Clintox**
```
Query: Is <SMILES> COC[C@@H](NC(C)=O)C(=O)NCC1=CC=CC=C1 </SMILES> toxic?

Response: <BOOLEAN> No </BOOLEAN>
```

**Property Prediction - HIV**
```
Query: Can <SMILES> CC1=CN(C2C=CCCC2O)C(=O)NC1=O </SMILES> serve as an inhibitor of HIV replication?

Response: <BOOLEAN> No </BOOLEAN>
```

**Property Prediction - SIDER**
```
Query: Are there any known side effects of <SMILES> CC1=CC(C)=C(NC(=O)CN(CC(=O)O)CC(=O)O)C(C)=C1Br </SMILES> affecting the heart?

Response: <BOOLEAN> No </BOOLEAN>
```

### Molecule Description

**Molecule Captioning**
```
Query: Describe this molecule: <SMILES> CCOC(=O)C1=CN=CN1[C@H](C)C1=CC=CC=C1 </SMILES>

Response: The molecule is an imidazole derivative with short-acting sedative, hypnotic, and general anesthetic properties. Etomidate appears to have gamma-aminobutyric acid (GABA) like effects, mediated through GABA-A receptor. The action enhances the inhibitory effect of GABA on the central nervous system by causing chloride channel opening events which leads to membrane hyperpolarization.
```

**Molecule Generation**

It is not required to use tags to wrap up the input description.

```
Query: Give me a molecule that satisfies the conditions outlined in the description: The molecule is a member of the class of tripyrroles that is a red-coloured pigment with antibiotic properties produced by Serratia marcescens. It has a role as an antimicrobial agent, a biological pigment, a bacterial metabolite, an apoptosis inducer and an antineoplastic agent. It is a tripyrrole, an aromatic ether and a ring assembly.

Response: Here is a potential molecule: <SMILES> CCCCCC1=C(C)NC(/C=C2\N=C(C3=CC=CN3)C=C2OC)=C1 </SMILES>
```

### Chemical Reaction

**Forward Synthesis**
```
Query: <SMILES> NC1=CC=C2OCOC2=C1.O=CO </SMILES> Based on the reactants and reagents given above, suggest a possible product.

Response: A possible product can be <SMILES> O=CNC1=CC=C2OCOC2=C1 </SMILES> .
```

**Retrosynthesis**
```
Query: Identify possible reactants that could have been used to create the specified product. <SMILES> CC1=CC=C(N)N=C1N </SMILES>

Response: <SMILES> CC(C#N)CCC#N.N </SMILES>
```


## Usage

Clone this repo to your machine, and `cd` to the folder.

### Generation

```python
from generation import LlaSMolGeneration

generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B')
generator.generate('Can you tell me the IUPAC name of <SMILES> C1CCOC1 </SMILES> ?')
```

**Note**: 
1. In the input query, please use corresponding tags to wrap up specific content. 
    - SMILES representation: `<SMILES> ... </SMILES>`
    - IUPAC name: `<IUPAC> ... </IUPAC>`
    
    Other tags may appear in models' responses:
    - Molecular formula: `<MOLFORMULA> ... </MOLFORMULA>`
    - Number: `<NUMBER> ... </NUMBER>`
    - Boolean: `<BOOLEAN> ... </BOOLEAN>`

    Please see the examples in [the above section](#tasks-and-examples).

2. The code would canonicalize SMILES string automatically, as long as it is wrapped in `<SMILES> ... </SMILES>`.


## To-Do Items
These will be uploaded soon:
- [x] Generation code.
- [ ] Detailed documentation.
- [ ] Evaluation code.
- [ ] More models' weights.

## Citation
If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.
```
@article{yu2024llasmol,
    title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
    author={Botao Yu and Frazier N. Baker and Ziqi Chen and Xia Ning and Huan Sun},
    journal={arXiv preprint arXiv:2402.09391},
    year={2024}
}
```