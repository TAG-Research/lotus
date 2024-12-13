Semantic Aggregation
======================

.. automodule:: lotus.sem_ops.sem_agg
    :members:
    :show-inheritance:


Examples
---------
.. code-block:: python
    import pandas as pd

    import lotus

    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    data = {
        "ArticleTitle": [
            "Advancements in Quantum Computing",
            "Climate Change and Renewable Energy",
            "The Rise of Artificial Intelligence",
            "A Journey into Deep Space Exploration"
        ],
        "ArticleContent": [
            """Quantum computing harnesses the properties of quantum mechanics 
            to perform computations at speeds unimaginable with classical machines. 
            As research and development progress, emerging quantum algorithms show 
            great promise in solving previously intractable problems.""",
            
            """Global temperatures continue to rise, and societies worldwide 
            are turning to renewable resources like solar and wind power to mitigate 
            climate change. The shift to green technology is expected to reshape 
            economies and significantly reduce carbon footprints.""",
            
            """Artificial Intelligence (AI) has grown rapidly, integrating 
            into various industries. Machine learning models now enable systems to 
            learn from massive datasets, improving efficiency and uncovering hidden 
            patterns. However, ethical concerns about privacy and bias must be addressed.""",
            
            """Deep space exploration aims to understand the cosmos beyond 
            our solar system. Recent missions focus on distant exoplanets, black holes, 
            and interstellar objects. Advancements in propulsion and life support systems 
            may one day enable human travel to far-off celestial bodies."""
        ]
    }

    df = pd.DataFrame(data)

    df = df.sem_agg("Provide a concise summary of all {ArticleContent} in a single paragraph, highlighting the key technological progress and its implications for the future.")
    print(df._output[0])

Output
"Recent technological advancements are reshaping various fields and have significant implications for the future. 
Quantum computing is emerging as a powerful tool capable of solving complex problems at unprecedented speeds, while the 
global shift towards renewable energy sources like solar and wind power aims to combat climate change and transform economies. 
In the realm of Artificial Intelligence, rapid growth and integration into industries are enhancing efficiency and revealing 
hidden data patterns, though ethical concerns regarding privacy and bias persist. Additionally, deep space exploration is 
advancing with missions targeting exoplanets and black holes, potentially paving the way for human travel beyond our solar 
system through improved propulsion and life support technologies."

