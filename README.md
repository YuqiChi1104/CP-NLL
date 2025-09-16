# CP-NLL
<h1>CP-NLL: Conformal Prediction for Noise Label Learning</h1>

<h2>Overview</h2>

<p>CP-NLL (Conformal Prediction for noise Label Learning) is a powerful framework designed to provide reliable predictions in scenarios where only partial labels are available. This repository contains scripts and resources to preprocess data, run models, generate scores, and visualize results.</p>

<h2>Project Structure</h2>
<ul>
<li><code>models/</code> - Directory containing model architectures for generating scores.</li>
<li><code>data/</code> - Directory is designated for storing the datasets required for the experiments.</li>
<li><code>cifar10-lt/</code> - Used to generate the long-tailed version of the CIFAR-10 dataset.</li>
<li><code>cifar100-lt/</code> - Used to generate the long-tailed version of the CIFAR-100 dataset.</li>
<li><code>requirements.txt</code> - Python dependencies.</li>
<li><code>results.py</code> - Script to generate the partial label prediction set size.</li>
<li><code>results.py</code> - Script to generate the partial label prediction set size.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Generate Scores</h3>

<p>Run the models inside the <code>models/</code> directory to generate scores. Instructions for running the models are provided inside the respective folders for all five models.</p>


<h3>Step 2: Generate Prediction Set Sizes</h3>

<p>Run the <code>results.py</code> script to generate the noisy label prediction set size:</p>

<pre><code>python results.py --base_path /Users/cp-nll/table_cifar_100_lt --epsilon 0.1 --partial_rate 0.2
</code></pre>

<p>The generated prediction sets can then be used to create plots.</p>
