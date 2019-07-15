======
EnsTop
======

EnsTop provides an ensemble based approach to topic modelling using pLSA. It makes
use of a high performance numba based pLSA implementation to run multiple
bootstrapped topic models in parallel, and then clusters the resulting outputs to
determine a set of stable topics. It can then refit the document vectors against
these topics embed documents into the stable topic space.

---------------
Why use EnsTop?
---------------

There are a number of advantages to using an ensemble approach to topic modelling.
The most obvious is that it produces better more stable topics. A close second,
however, is that, by making use of HDBSCAN for clustering topics, it can learn a
"natural" number of topics. That is, while the user needs to specify an estimated
number of topics, the *actual* number of topics produced will be determined by how
many stable topics are produced over many bootstrapped runs. In practice this can
either be more, or less, than the estimated number of topics.

Despite all of these extra features the ensemble topic approach is still very
efficient, especially in multi-core environments (due the the embarrassingly parallel
nature of the ensemble). A run with a reasonable size ensemble can be completed in
around the same time it might take to fit an LDA model, and usually produces superior
quality results.

In addition to this EnsTop comes with a pLSA implementation that can be used
standalone (and not as part of an ensemble). So if all you are loosing for is a good
fast pLSA implementation (that can run considerably faster than many LDA
implementations) then EnsTop is the library for you.

-----------------
How to use EnsTop
-----------------

EnsTop follows the sklearn API (and inherits from sklearn base classes), so if you
use sklearn for LDA or NMF then you already know how to use Enstop. General usage is
very straightforward. The following example uses EnsTop to model topics from the
classic 20-Newsgroups dataset, using sklearn's CountVectorizer to generate the
required count matrix.

.. code:: python

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from enstop import EnsembleTopics

    news = fetch_20newsgroups(subset='all')
    data = CountVectorizer().fit_transform(news.data)

    model = EnsembleTopics(n_components=20).fit(data)
    topics = model.components_
    doc_vectors = model.embedding_


---------------
How to use pLSA
---------------

EnsTop also provides a simple to use but fast and effective pLSA implementation out
of the box. As with the ensemble topic modeller it follows the sklearn API, and usage
is very similar.

.. code:: python

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from enstop import PLSA

    news = fetch_20newsgroups(subset='all')
    data = CountVectorizer().fit_transform(news.data)

    model = PLSA(n_components=20).fit(data)
    topics = model.components_
    doc_vectors = model.embedding_


------------
Installation
------------

The easiest way to install EnsTop is via pip

.. code:: bash

    pip install enstop

To manually install this package:

.. code:: bash

    wget https://github.com/lmcinnes/enstop/archive/master.zip
    unzip master.zip
    rm master.zip
    cd enstop-master
    python setup.py install

----------------
Help and Support
----------------

Some basic example notebooks are available `here <./notebooks>`_.

Documentation is coming. This project is still very young. If you need help, or have
problems please `open an issue <https://github.com/lmcinnes/enstop/issues/new>`_
and I will try to provide any help and guidance that I can. Please also check
the docstrings on the code, which provide some descriptions of the parameters.


-------
License
-------

The EnsTop package is 2-clause BSD licensed.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. To contribute please `fork the project <https://github.com/lmcinnes/enstop/issues#fork-destination-box>`_ make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.
