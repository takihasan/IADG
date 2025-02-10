# IADG

# Instance-Aware Test-Time Adaptation for Domain Generalization
**Taki Hasan Rafi**, Serbeter Karlo, Amit Agarwal, Hitesh Patel, Bhargava Kumar, Dong-Kyu Chae
Hanyang University and Oracle USA collaboration

The paper has been accepted by **DASFAA 2025**.
## Abstract

Domain generalization (DG) aims to enhance the generalization capability of models to unseen target distributions by leveraging multiple source distributions. This paper focuses on robust test-time adaptation (TTA) based DG that is mostly needed when the model experiences new unseen domains while testing. We consider two practical challenges under domain shifts: (1) existing DG methods mostly rely on domain-specific information and do not explicitly utilize class-specific information. Therefore, these approaches ignore mixed features, which are both class and domain-specific, thus resulting in the loss of useful information; (2) while existing TTA methods explicitly require a memory bank of test time samples, which is computationally complex and impractical in many applications. To overcome these limitations, we propose a new framework called IADG that utilizes class-specific information along with domain-specific information to ensure robust generalization. Our method exploits disentangled features by pulling class-relevant features to increase diversified negative pairs, facilitating flawless integration of class and domain-specific features. To leverage high-confidence samples during testing, we introduce a novel confidence-guided low-risk instance TTA approach that only considers one unlabeled sample during inference and therefore does not require a dedicated memory bank. Extensive evaluations on five public benchmarks consistently demonstrate the superior performance of our approach over the state-of-the-art. 
