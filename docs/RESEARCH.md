# Research notes

Literature behind the design (searched November 2026). Numbers are as reported by the authors.

## Key findings

1. **Random splits overstate accuracy.** Lab-trained models fall to about 26–30 % on field photos [1]. A fine-tuned
   ResNet-50 lost 67.7 points while staying about 80 % confident [2]. Rice classifiers collapse across Bangladeshi
   datasets; strong augmentation helps and AdaBN hurts [3].
2. **Frozen foundation features carry over best.** A DINOv2 linear probe gives +23 points over EfficientNet-B0 on field photos,
   and fine-tuning the backbone hurts. Distilling it into a 4M-parameter CNN runs at 33 ms on a CPU [1]. Fusing diverse
   frozen backbones gives 80.2 % on PlantWild [4].
3. **Distillation:** plain logit KD is the best configuration (+3.1 points) [5]. Ensemble → KD → INT8 keeps 97.5 % accuracy
   at 1.46 MB [6].
4. **Unknown diseases:** AM-Softmax with plant-domain pretraining [7]; max-logit under different fine-tuning strategies [8];
   a knowledge ensemble cuts false positives at 95 % detection from 15.4 % to 0.71 % [9]; open-world incremental learning with replay [10];
   a prototype library that grows without retraining [11]; open-set evaluation on Paddy Doctor [12].

## References

1. Nguyen et al. (2026). From laboratory to field: Frozen foundation-model features toward robust plant disease recognition. *Smart Agricultural Technology*.
2. Xiang, Shi & Zhu (2026). Quantifying the reliability gap in cross-domain plant disease classification. *Frontiers in Plant Science*.
3. Paul (2026). Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Classifiers. *Zenodo*.
4. El Karch et al. (2026). Backbone diversity beats text supervision: frozen multi-foundation model fusion for in-the-wild plant disease recognition. *Frontiers in Plant Science*.
5. Zhou et al. (2026). Cross-Architecture Knowledge Distillation from a Vision Foundation Model to a Lightweight Visual State Space Model for Tea Leaf Disease Classification. *arXiv*.
6. Hasan et al. (2025). Deployable Deep Learning for Cross-Domain Plant Leaf Disease Detection via Ensemble Learning, Knowledge Distillation, and Quantization. *IEEE Access*.
7. Meng et al. (2023). Known and unknown class recognition on plant species and diseases. *Computers and Electronics in Agriculture*.
8. Dong et al. (2024). The impact of fine-tuning paradigms on unknown plant diseases recognition. *Scientific Reports*.
9. Dong et al. (2025). Enhancing anomaly detection in plant disease recognition with knowledge ensemble. *Frontiers in Plant Science*.
10. Dong et al. (2023). A New Deep Learning-based Dynamic Paradigm Towards Open-World Plant Disease Detection. *Frontiers in Plant Science*.
11. Yuan et al. (2026). A knowledge-driven unified framework for plant disease classification and severity grading via domain adaptation. *Plant Phenomics*.
12. Yan et al. (2026). PAG-MSBNet: Prototype-aware gated multi-scale branch network for open-set crop disease recognition. *Ecological Informatics*.
13. Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv*.

Dataset citations are in `configs/datasets.yaml`.
