Setting Configurations
=======================

The Settings module is a central configuration system for managing application-wide settings. 
It ensures consistent and thread-safe access to configurations, allowing settings to be dynamically 
adjusted and temporarily overridden within specific contexts. In most examples seen, we have 
used the settings to configured our LM.

Using the Settings module
--------------------------
.. code-block:: python
    
    from lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

Configurable Parameters
--------------------------

1. enable_cache: 
    * Description: Enables or Disables cahcing mechanisms
    * Default: False
.. code-block:: python

    settings.configure(enable_cache=True)

2. cascade_IS_weight: 
    * Description: Specifies the weight for importance Sampling in cascade Operators
    * Default: 0.5
.. code-block:: python

    settings.configure(cascade_IS_weight=0.8)

3. cascade_num_calibration_quantiles:
    * Description: Number of quantiles used for calibrating sem_filter
    * Defualt: 50
.. code-block:: python

    settings.configure(cascade_num_calibration_quantiles=100)

4. min_join_cascade_size:
    * Description: Minimum size of qa join cascade to trigger additional Processing
    * Default: 100
.. code-block:: python 

    settings.configure(min_join_cascade_size=200)

5. cascade_IS_max_sample_range:
    * DescriptionL maximum range for sampling during cascade IS Operations
    * Default: 250
.. code-block:: python

    settings,configure(cascade_IS_max_sample_range= 500)

6. cascade_IS_random_seed:
    * Description: Seed value for randomization in casde IS. Use None for non-deterministic behavior
    * Default: None
.. code-block:: python
    
    settings.configure(cascade_IS_random_seed=42)