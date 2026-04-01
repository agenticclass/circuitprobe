"""
Calibration datasets for CircuitProbe.

Contains mixed reasoning and general text examples.
These are used to collect activation statistics that distinguish
reasoning-critical layers from other layers.
"""

REASONING_EXAMPLES = [
    # Math reasoning
    "If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance traveled?",
    "A store offers a 20% discount on a $150 item, then charges 8% sales tax. What is the final price?",
    "If 3x + 7 = 22, what is the value of x?",
    "A rectangular garden is 12 meters long and 8 meters wide. What is the length of the diagonal?",
    "If you invest $1000 at 5% annual interest compounded monthly, how much will you have after 2 years?",
    "Three friends split a bill of $87.30 equally. How much does each person pay?",
    "A car depreciates by 15% each year. If it costs $30,000 new, what is it worth after 3 years?",
    "What is the probability of rolling a sum of 7 with two fair dice?",
    "If a recipe requires 2/3 cup of flour and you want to make 1.5 times the recipe, how much flour do you need?",
    "A pool fills at 3 gallons per minute and drains at 1 gallon per minute. How long to fill a 200 gallon pool?",

    # Logical reasoning
    "All cats are mammals. All mammals are animals. Therefore, all cats are animals. Is this valid?",
    "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
    "Alice is taller than Bob. Bob is taller than Charlie. Who is the shortest?",
    "Every student in the class passed the exam. John is a student in the class. What can we conclude?",
    "If P implies Q, and Q implies R, and P is true, what can we conclude about R?",
    "None of the birds in this aviary can fly. Tweety is a bird in this aviary. Can Tweety fly?",
    "Some doctors are runners. All runners are athletes. Can we conclude that some doctors are athletes?",
    "If today is not Monday, then the store is open. The store is closed. What day is it?",

    # Causal reasoning
    "A plant in a dark room wilts after two weeks. A plant in sunlight thrives. What caused the first plant to wilt?",
    "After a new speed limit was introduced, accidents decreased by 30%. What likely caused the decrease?",
    "A student studied for 5 hours and got an A. Another studied for 30 minutes and got a C. What explains the difference?",

    # Multi-step reasoning
    "John has twice as many apples as Mary. Mary has three more apples than Tom. Tom has 4 apples. How many apples does John have?",
    "A shirt is marked down 25% from $80. Then an additional 10% off the sale price. What is the final price?",
    "If the first train leaves at 8am going 50mph and the second leaves at 9am going 75mph, when does the second catch up?",
    "There are 5 houses in a row. The red house is to the left of the blue house. The green house is between them. The yellow house is at the far left. What color is the house at the far right?",
]

GENERAL_EXAMPLES = [
    # Factual / descriptive
    "The Amazon rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers.",
    "Python is a popular programming language known for its clean syntax and extensive library ecosystem.",
    "The human body contains approximately 206 bones and over 600 muscles.",
    "Tokyo is the most populous metropolitan area in the world with over 37 million people.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Great Wall of China spans over 21,000 kilometers and was built over many centuries.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The periodic table organizes chemical elements by atomic number and electron configuration.",
    "Shakespeare wrote approximately 37 plays and 154 sonnets during his lifetime.",

    # Conversational / instructions
    "Can you help me write a thank you note for a job interview I had yesterday?",
    "I need a recipe for chocolate chip cookies that serves about 24 people.",
    "What are some good exercises for someone who sits at a desk all day?",
    "Please summarize the main differences between TCP and UDP protocols.",
    "How do I set up a virtual environment in Python?",

    # Creative / open-ended
    "Write a short paragraph describing a sunset over the ocean.",
    "What would happen if humans could photosynthesize like plants?",
    "Describe the taste of chocolate to someone who has never had it.",
    "If you could have dinner with any historical figure, who would you choose and why?",
    "Tell me about the most interesting thing you know about octopuses.",

    # Technical but not reasoning-heavy
    "Git is a distributed version control system created by Linus Torvalds in 2005.",
    "Machine learning models are typically trained using gradient descent optimization.",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for communication.",
    "Docker containers package applications with their dependencies for consistent deployment.",
    "SQL databases organize data into tables with rows and columns, using structured query language for data manipulation.",
]


def get_calibration_set(n_reasoning=25, n_general=25):
    """
    Get a mixed calibration set of reasoning and general text examples.

    Returns:
        Tuple of (texts, labels) where labels are 'reasoning' or 'general'.
    """
    r = REASONING_EXAMPLES[:n_reasoning]
    g = GENERAL_EXAMPLES[:n_general]

    texts = r + g
    labels = ["reasoning"] * len(r) + ["general"] * len(g)

    return texts, labels


def get_contrastive_sets():
    """
    Get separate reasoning and general sets for contrastive analysis.

    Returns:
        Tuple of (reasoning_texts, general_texts).
    """
    return REASONING_EXAMPLES, GENERAL_EXAMPLES


# Multilingual calibration sets for Section 8 experiments
HINDI_REASONING = [
    "एक ट्रेन 60 किमी/घंटा की रफ्तार से 2.5 घंटे चलती है, फिर 80 किमी/घंटा से 1.5 घंटे। कुल दूरी क्या है?",
    "एक दुकान $150 की वस्तु पर 20% छूट देती है, फिर 8% बिक्री कर लगाती है। अंतिम कीमत क्या है?",
    "अगर 3x + 7 = 22, तो x का मान क्या है?",
    "जॉन के पास मैरी से दोगुने सेब हैं। मैरी के पास टॉम से 3 अधिक हैं। टॉम के पास 4 सेब हैं। जॉन के पास कितने हैं?",
    "सभी बिल्लियाँ स्तनधारी हैं। सभी स्तनधारी जानवर हैं। क्या सभी बिल्लियाँ जानवर हैं?",
    "अगर बारिश होती है तो ज़मीन गीली हो जाती है। ज़मीन गीली है। क्या हम कह सकते हैं कि बारिश हुई?",
    "एलिस बॉब से लंबी है। बॉब चार्ली से लंबा है। सबसे छोटा कौन है?",
    "एक शर्ट $80 से 25% कम है। फिर बिक्री मूल्य से 10% अतिरिक्त छूट। अंतिम कीमत क्या है?",
    "एक पौधा अंधेरे कमरे में दो सप्ताह बाद मुरझा जाता है। धूप में पौधा फलता-फूलता है। पहले पौधे के मुरझाने का कारण?",
    "अगर P का अर्थ Q है, और Q का अर्थ R है, और P सत्य है, तो R के बारे में क्या निष्कर्ष?",
]

CHINESE_REASONING = [
    "一列火车以60公里/小时的速度行驶2.5小时，然后以80公里/小时行驶1.5小时。总距离是多少？",
    "一家商店对150美元的商品打八折，然后收取8%的销售税。最终价格是多少？",
    "如果3x + 7 = 22，x的值是什么？",
    "约翰的苹果是玛丽的两倍。玛丽比汤姆多3个。汤姆有4个苹果。约翰有多少个？",
    "所有猫都是哺乳动物。所有哺乳动物都是动物。所以所有猫都是动物。这个推理正确吗？",
    "如果下雨，地面会湿。地面是湿的。我们能得出下过雨的结论吗？",
    "爱丽丝比鲍勃高。鲍勃比查理高。谁最矮？",
    "一件衬衫从80美元打七五折。然后在售价基础上再打九折。最终价格是多少？",
    "一棵植物在黑暗的房间里两周后枯萎了。在阳光下的植物茁壮成长。第一棵植物枯萎的原因是什么？",
    "如果P蕴含Q，Q蕴含R，且P为真，那么关于R可以得出什么结论？",
]

FRENCH_REASONING = [
    "Un train roule à 60 km/h pendant 2,5 heures, puis à 80 km/h pendant 1,5 heures. Quelle est la distance totale parcourue ?",
    "Un magasin offre une réduction de 20% sur un article à 150$, puis applique une taxe de 8%. Quel est le prix final ?",
    "Si 3x + 7 = 22, quelle est la valeur de x ?",
    "John a deux fois plus de pommes que Mary. Mary a 3 pommes de plus que Tom. Tom a 4 pommes. Combien John en a-t-il ?",
    "Tous les chats sont des mammifères. Tous les mammifères sont des animaux. Donc tous les chats sont des animaux. Est-ce valide ?",
    "S'il pleut, le sol est mouillé. Le sol est mouillé. Peut-on conclure qu'il a plu ?",
    "Alice est plus grande que Bob. Bob est plus grand que Charlie. Qui est le plus petit ?",
    "Une chemise est soldée à 25% de réduction sur 80$. Puis 10% de réduction supplémentaire. Quel est le prix final ?",
    "Une plante dans une pièce sombre se fane après deux semaines. Une plante au soleil prospère. Qu'est-ce qui a causé le flétrissement ?",
    "Si P implique Q, et Q implique R, et P est vrai, que peut-on conclure sur R ?",
]


def get_multilingual_sets():
    """Get reasoning calibration sets in multiple languages."""
    return {
        "english": REASONING_EXAMPLES[:10],
        "hindi": HINDI_REASONING,
        "chinese": CHINESE_REASONING,
        "french": FRENCH_REASONING,
    }
