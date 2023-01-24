# HistoMIL
A modified CLAM/MIL model for large histopathology WSIs. The model breaks down to these components:

<ol>
  <li>A VGG16 embedding backbone, previously trained on the WSI patches with labels propagated from the slide level, reducing each patch to a 1024 feature vector. </li>
  <li>All embeddings from a slide are aggregated into a larger feature vector and passed to the attention model</li>
  <li>The attention modell (CLAM/MIL) aggregates the patch-level information to the slide level, as well as calculating an attention score for each patch/li>
  <li>A slide level prediction is made</li>
  <li>A heatmap is created for the slide</li>
</ol>

![alt text](https://github.com/[AmayaGS]/[HistoMIL]/blob/[branch]/image.jpg?raw=true)

<p>References:<br>
https://github.com/mahmoodlab/CLAM#readme <br>
https://github.com/AMLab-Amsterdam/AttentionDeepMIL</p>
