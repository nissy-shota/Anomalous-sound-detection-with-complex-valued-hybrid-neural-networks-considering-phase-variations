# Anomalous sound detection with complex-valued hybrid neural networks considering phase variations

[Paper](https://www.ieice.org/publications/ken/summary.php?contribution_id=123914&society_cd=ESSNLS&ken_id=EA&year=2023&presen_date=2023-03-01&schedule_id=7826&lang=en&expandable=0): Anomalous sound detection with complex-valued hybrid neural networks considering phase variations  
Author: Shota Nishiyama, Akira Tamamori  
Affiliation, Organization: Aichi Institute of Technology Graduate School of Business Administration and Computer Science (AIT)  
keyword: Anomalous sound detection, complex-valued neural networks, phase variations  
paper info: EA2022-106,SIP2022-150,SP2022-70  
date of issue: 2023-02-21 (EA, SIP, SP)  

## Abstract

Anomalous sound detection is the task of identifying whether an incoming mechanical sound is normal or anomalous. Since anomalous sounds occur infrequently and are highly diverse, it is treated as a problem of detecting anomalous sounds from normal sounds only. The acoustic features used as input to most anomalous sound detection models are mel-spectrogram. However, the phase variation is lost when the complex-spectrogram obtained by Fourier transforming the sound waveform is converted to the mel-spectrogram. In this study, we compare anomalous sound detection methods using complex-valued neural networks and real-valued neural networks to demonstrate the usefulness of phase variation. As a result of the comparison, there existed machine sounds for which phase variation was valuable and machine sounds for which it was not valuable. In this study, we propose a complex-valued hybrid neural network that combines a complex-valued module that preserves the structure of complex values and a real-valued module that takes mel-spectrogram as input for all feature extraction operations in which complex-spectrogram can be input in order to take phase variation into account. We propose a complex-valued hybrid neural network that combines a complex-valued structure-preserving module and a real-valued module that takes the mel-spectrogram as input for all feature extraction operations. Experiments verified the effectiveness of the proposed method on anomalous sound detection for multi-channel sound in the ToyADMOS dataset. Experimental results showed that the proposed method improved the average AUC of all machine sounds by around 3% compared to both complex-valued and real-valued neural networks.

## Usege

```bash
cd environments/gpu
docker compose up -d
docker compose exec complex-toyadmos bash
poetry install
poetry run bash all_features_run.sh
```

> [!NOTE]  
plase rewrite your environment -> `configures/experiments/default.yaml`

## Citation

```
@techreport{weko_224446_1,
   author	 = "西山,翔大 and 玉森,聡",
   title	 = "位相変動を考慮した複素数値ハイブリッドニューラルワークによる異常音検知",
   year 	 = "2023",
   institution	 = "愛知工業大学大学院経営情報科学研究科, 愛知工業大学大学院経営情報科学研究科",
   number	 = "49",
   month	 = "feb"
}
```