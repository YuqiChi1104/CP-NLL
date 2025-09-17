# Cross-Entropy

<b>Experiments</b>\
First, please create a folder named <i>checkpoint</i> to store the results.\
<code>mkdir checkpoint</code>\
Next, run \
<code>python Train_{dataset_name}.py --data_path <i>path-to-your-data</i></code>

<h2>How to generate scores</h2>

<h3>Cifar10</h3>

<pre><code>python /your/code/file/models/CE/generate_result.py
</code></pre>

<h3>Cifar100</h3>

<pre><code>python /your/code/file/models/CE/generate_result_cifar100.py
</code></pre>

<h3>Cifar10lt</h3>

<pre><code>python /your/code/file/models/CE/generate_result_lt.py
</code></pre>

<h3>Cifar100lt</h3>

<pre><code>python /your/code/file/models/CE/generate_result_cifar100lt.py
</code></pre>

<b>Cite DivideMix</b>\
If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{
    li2020dividemix,
    title={DivideMix: Learning with Noisy Labels as Semi-supervised Learning},
    author={Junnan Li and Richard Socher and Steven C.H. Hoi},
    booktitle={International Conference on Learning Representations},
    year={2020},
}</pre>

<b>License</b>\
This project is licensed under the terms of the MIT license.
