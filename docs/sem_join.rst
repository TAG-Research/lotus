Semantic Join
=================

.. automodule:: lotus.sem_ops.sem_join
    :members:
    :show-inheritance:

Join Example
--------------
.. code-block:: python
    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)
    data = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
            "Operating Systems",
            "Food Science",
            "Compilers",
            "Intro to computer science",
        ]
    }

    data2 = {"Skill": ["Math", "Computer Science"]}

    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    join_instruction = "Taking {Course Name:left} will help me learn {Skill:right}"
    res = df1.sem_join(df2, join_instruction)
    print(res)

Output
+---+----------------------------+-------------------+
|   |      Course Name           |       Skill       |
+---+----------------------------+-------------------+                
| 1 |  Riemannian Geometry       |       Math        |
| 2 |   Operating Systems        |  Computer Science |
| 4 |      Compilers             |  Computer Science |
| 5 | Intro to computer science  |  Computer Science |
+---+----------------------------+-------------------+


Join Cascade Example
----------------------
.. code-block:: python
    import pandas as pd

    import lotus
    from lotus.models import LM, SentenceTransformersRM
    from lotus.types import CascadeArgs

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(lm=lm, rm=rm)
    data = {
        "Course Name": [
            "Digital Design and Integrated Circuits",
            "Data Structures and Algorithms",
            "The History of Art",
            "Natural Language Processing",
        ]
    }

    skills = [
        "Math", "Computer Science", "Management", "Creative Writing", "Data Analysis", "Machine Learning", "Project Management",
        "Problem Solving", "Singing", "Critical Thinking", "Public Speaking", "Teamwork", "Adaptability", "Programming",
        "Leadership", "Time Management", "Negotiation", "Decision Making", "Networking", "Painting",
        "Customer Service", "Marketing", "Graphic Design", "Nursery", "SEO", "Content Creation", "Video Editing", "Sales",
        "Financial Analysis", "Accounting", "Event Planning", "Foreign Languages", "Software Development", "Cybersecurity",
        "Social Media Management", "Photography", "Writing & Editing", "Technical Support", "Database Management", "Web Development",
        "Business Strategy", "Operations Management", "UI/UX Design", "Reinforcement Learning", "Data Visualization",
        "Product Management", "Cloud Computing", "Agile Methodology", "Blockchain", "IT Support", "Legal Research", "Supply Chain Management",
        "Copywriting", "Human Resources", "Quality Assurance", "Medical Research", "Healthcare Management", "Sports Coaching",
        "Editing & Proofreading", "Legal Writing", "Human Anatomy", "Chemistry", "Physics", "Biology",
        "Psychology", "Sociology", "Anthropology", "Political Science", "Public Relations", "Fashion Design", "Interior Design",
        "Automotive Repair", "Plumbing", "Carpentry", "Electrical Work", "Welding", "Electronics", "Hardware Engineering",
        "Circuit Design", "Robotics", "Environmental Science", "Marine Biology", "Urban Planning", "Geography",
        "Agricultural Science", "Animal Care", "Veterinary Science", "Zoology", "Ecology", "Botany", "Landscape Design",
        "Baking & Pastry", "Culinary Arts", "Bartending", "Nutrition", "Dietary Planning", "Physical Training", "Yoga",
    ]
    data2 = pd.DataFrame({"Skill": skills})


    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    join_instruction = "By taking {Course Name:left} I will learn {Skill:right}"

    cascade_args = CascadeArgs(recall_target=0.7, precision_target=0.7)
    res, stats = df1.sem_join(df2, join_instruction, cascade_args=cascade_args, return_stats=True)


    print(f"Joined {df1.shape[0]} rows from df1 with {df2.shape[0]} rows from df2")
    print(f"    Join cascade took {stats['join_resolved_by_large_model']} LM calls")
    print(f"    Helper resolved {stats['join_resolved_by_helper_model']} LM calls")
    print(f"Join cascade used {stats['total_LM_calls']} LM calls in total")
    print(f"Naive join would require {df1.shape[0]*df2.shape[0]} LM calls")
    print(res)

Output
+---+----------------------------------------+----------------------+
|   |            Course Name                 |        Skill         |
+---+----------------------------------------+----------------------+
| 0 | Digital Design and Integrated Circuits | Circuit Design       |
| 3 | Natural Language Processing            | Machine Learning     |
| 1 | Data Structures and Algorithms         | Computer Science     |
| 0 | Digital Design and Integrated Circuits | Electronics          |
| 0 | Digital Design and Integrated Circuits | Hardware Engineering |
+---+----------------------------------------+----------------------+