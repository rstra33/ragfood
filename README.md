Here’s a clear, beginner-friendly `README.md` for your RAG project, designed to explain what it does, how it works, and how someone can run it from scratch.

---

## 📄 `README.md`

````markdown
# 🧠 RAG-Food: Simple Retrieval-Augmented Generation with ChromaDB + Ollama

This is a **minimal working RAG (Retrieval-Augmented Generation)** demo using:

- ✅ Local LLM via [Ollama](https://ollama.com/)
- ✅ Local embeddings via `mxbai-embed-large`
- ✅ [ChromaDB](https://www.trychroma.com/) as the vector database
- ✅ A simple food dataset in JSON (Indian foods, fruits, etc.)

---

## 🎯 What This Does

This app allows you to ask questions like:

- “Which Indian dish uses chickpeas?”
- “What dessert is made from milk and soaked in syrup?”
- “What is masala dosa made of?”

It **does not rely on the LLM’s built-in memory**. Instead, it:

1. **Embeds your custom text data** (about food) using `mxbai-embed-large`
2. Stores those embeddings in **ChromaDB**
3. For any question, it:
   - Embeds your question
   - Finds relevant context via similarity search
   - Passes that context + question to a local LLM (`llama3.2`)
4. Returns a natural-language answer grounded in your data.

---

## 📦 Requirements

### ✅ Software

- Python 3.8+
- Ollama installed and running locally
- ChromaDB installed

### ✅ Ollama Models Needed

Run these in your terminal to install them:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
````

> Make sure `ollama` is running in the background. You can test it with:
>
> ```bash
> ollama run llama3.2
> ```

---

## 🛠️ Installation & Setup

### 1. Clone or download this repo

```bash
git clone https://github.com/yourname/rag-food
cd rag-food
```

### 2. Install Python dependencies

```bash
pip install chromadb requests
```

### 3. Run the RAG app

```bash
python rag_run.py
```

If it's the first time, it will:

* Create `foods.json` if missing
* Generate embeddings for all food items
* Load them into ChromaDB
* Run a few example questions

---

## 📁 File Structure

```
rag-food/
├── rag_run.py       # Main app script
├── foods.json       # Food knowledge base (created if missing)
├── README.md        # This file
```

---

## 🧠 How It Works (Step-by-Step)

1. **Data** is loaded from `foods.json`
2. Each entry is embedded using Ollama's `mxbai-embed-large`
3. Embeddings are stored in ChromaDB
4. When you ask a question:

   * The question is embedded
   * The top 1–2 most relevant chunks are retrieved
   * The context + question is passed to `llama3.2`
   * The model answers using that info only

---

## 🔍 Try Custom Questions

You can update `rag_run.py` to include your own questions like:

```python
print(rag_query("What is tandoori chicken?"))
print(rag_query("Which foods are spicy and vegetarian?"))
```

---

## 🚀 Next Ideas

* Swap in larger datasets (Wikipedia articles, recipes, PDFs)
* Add a web UI with Gradio or Flask
* Cache embeddings to avoid reprocessing on every run

---

## 👨‍🍳 Credits

Made by Callum using:

* [Ollama](https://ollama.com)
* [ChromaDB](https://www.trychroma.com)
* [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large)
* Indian food inspiration 🍛






## Project Customizations
Name: Remi Strachan

Enhanced Food Database:
This project has been customized with comprehensive food entries focusing on detailed nutritional information, cooking methods, cultural significance, and dietary classifications.
15+ New Items Added (Entries 76-90).

Food Items Added:

(5 dishes from France: ID 76-80)
1. **Beef Bourguignon** — French Burgundy stew with beef braised in red wine for 2-3 hours; rich in protein and B vitamins
2. **Coq au Vin** — Rustic French chicken braised in wine (45-60 min); infused with complex flavors from wine reduction
3. **Ratatouille** — Provençal vegetable stew with eggplant, zucchini, tomatoes; rich in vitamins C & K, naturally low-calorie
4. **Bouillabaisse** — Legendary Mediterranean fish stew with saffron broth; UNESCO intangible cultural heritage
5. **Croissant** — Laminated pastry with hundreds of butter layers; symbol of French breakfast culture

(5 Healthy Foods: ID 81-85)
1. **Spinach** — Nutrient-dense leafy green with iron, vitamins K & A, and antioxidants; supports eye health, bone strength, and cardiovascular function
2. **Blueberries** — Antioxidant-rich berries with anthocyanins and vitamin C; improves cognitive function with anti-inflammatory benefits
3. **Salmon** — Excellent source of omega-3 fatty acids (EPA/DHA), protein, and vitamin D; supports heart and brain health
4. **Almonds** — Tree nuts rich in monounsaturated fats, plant-based protein, and vitamin E; promotes healthy cholesterol and sustained energy
5. **Broccoli** — Cruciferous vegetable with vitamin C, fiber, and sulforaphane; supports immune function and detoxification

(5 International Dishes: ID 86-90)
1. **Paella** — Legendary Valencian rice dish with saffron; UNESCO-recognized cultural symbol with signature socarrat (crispy bottom)
2. **Risotto** — Creamy Northern Italian rice with Arborio rice and continuous stirring (16-20 min); multiple regional variations
3. **Chilaquiles** — Traditional Mexican breakfast with fried tortilla strips simmered in salsa; pre-Hispanic origin with vegetarian options
4. **Barbecue Ribs** — Quintessential American slow-cooked dish (3-6 hours); regional variations (Carolina, Kansas City, Texas, Memphis)
5. **Jollof Rice** — Iconic West African one-pot dish with tomato sauce and crispy rice layer; culturally significant celebration food


Each of the above entries contained:
- ✅ 50+ word comprehensive descriptions per entry
- ✅ Detailed preparation methods with specific temperatures and cooking times
- ✅ Exact nutritional information (nutrient amounts, calorie counts, percentages)
- ✅ Cultural significance, regional variations, and historical context
- ✅ Dietary classifications (vegetarian/vegan/gluten-free/dairy-free indicators)

All entries are fully indexed in ChromaDB for semantic search and RAG-based question answering.

(Installation and setup done during week 2 zoom meeting)


**Sample Queries:**

(1)
User: What is beef burgundy?
Response: Beef Bourguignon. 
(unsatisfying answer, added more detail into description)

(2)
User: What is Ratatouille?
Response: A traditional Provençal vegetable stew originating from Nice, made by layering and slow-cooking eggplant, zucchini, bell peppers, tomatoes, onions, and garlic with Provençal herbs (basil, thyme, oregano) and olive oil for 30-45 minutes until vegetables are tender and flavors meld.

(3)
User: What is Risotto?
Response: Risotto is a creamy Northern Italian rice dish made exclusively with Arborio or Carnaroli rice, typically containing butter and Parmigiano-Reggiano cheese for richness, and various vegetable nutrients depending on the variation.

(4)
User: Which foods are high in protein?
Response:According to the context:
1. Salmon - 25g of protein per 100g
2. Almonds - 6g of protein per ounce

(5)
User: Tell me about french food.
Response: French cuisine! Known for its rich flavors, intricate preparations, and high-quality ingredients, French food is a culinary delight that celebrates the art of cooking. From hearty stews like Coq au vin to elegant dishes like Bouillabaisse, French cuisine has something to offer every palate.
At its core, French cuisine is about using fresh, seasonal ingredients to create dishes that are both nourishing and delicious. Many classic French dishes, such as Ratatouille and Tarte Tatin, showcase the country's love of vegetables, fruits, and herbs, which are often combined with rich sauces and flavorful oils.
French cooking also places a strong emphasis on technique and presentation. Dishes like Coq au vin and Beef Bourguignon require slow-cooking methods to achieve tender, fall-apart meat, while others, like Escargots and Ratatouille, rely on precise layering and seasoning techniques to create harmonious flavors.
Some popular French dishes include:
* Coq au vin: a rustic chicken dish braised in red wine with bacon lardons
* Beef Bourguignon: a classic Burgundy stew featuring chuck or braising cuts of beef
* Ratatouille: a colorful Provençal vegetable stew originating from Nice
* Escargots: snails cooked in garlic butter and parsley
When it comes to dietary restrictions, French cuisine can be adapted to suit various needs. Many traditional dishes are naturally gluten-free or vegan-friendly, while others may require substitutions to accommodate dietary preferences.
Overall, French food is a celebration of the country's culinary heritage and its love of good food, company, and conviviality. Whether you're in the mood for a comforting stew or an elegant dinner party dish, French cuisine has something to offer every taste and preference.

(6)
User: What vegan options are available?
Response: Based on the context provided:
1. Spinach is a nutrient-dense leafy green vegetable that can be consumed raw, lightly sautéed in garlic and olive oil, or blended into smoothies and soups.
2. Almonds (when blanched) retain maximum nutrients and are rich in monounsaturated fats, plant-based protein, vitamin E, magnesium, and fiber, making them an excellent option for a vegan diet, particularly when considering keto and paleo diets.

(7)
User: What options are vegetarian?
Response: The options that are vegetarian from the given context are:
1. Tabbouleh
2. Ratatouille
3. Broccoli
All three options can be adapted to a vegetarian diet, with no animal products or by-products included in their preparation.

(8)
User: What foods can be grilled? 
Response: Barbecue ribs.

**Personal Reflection on RAG learning experience:**
This week marked my introduction to Retrieval-Augmented Generation (RAG) — a technique that fundamentally changes how AI systems handle knowledge. Rather 
than relying solely on what a language model memorised during training, RAG enables AI to actively fetch relevant facts from a designated knowledge base at query time. 
One of the most clarifying moments this week was understanding how RAG differs from conventional keyword matching, such as what Google has historically used. 
Keyword search looks for literal term overlap — if you search for "chicken recipes," it finds pages containing those exact words. RAG, by contrast, works through semantic understanding. It comprehends meaning and context, not just character patterns.
This means a RAG system can match a query like "what should I eat for protein?" to a document about "high-leucine foods" — even though those words share nothing in common. That semantic bridging is what makes RAG genuinely intelligent, and it is enabled by the concept I found most fascinating this week: vector embeddings.
Leaning about vector embeddings shifted my thinking about AI from viewing it as 
a static oracle to understanding it as a dynamic retrieval system that can be 
pointed at any knowledge source. Once I understood that meaning can be encoded 
as geometry, semantic search went from feeling like magic to feeling like 
mathematics, which as a mathematics university student, I found extremely 
captivating.
The ragfood project provided a strong foundation; I am now thinking about what more complex, real-world datasets would surface as challenges in the pipeline and going forward, I want to experiment with different embedding models to compare retrieval quality.








