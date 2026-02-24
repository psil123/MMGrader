from prompt import *
from MentalModel import *
from models import *
import json
if __name__ == "__main__":
    model = Molmo(CL_SCORE_PROMPT=CL_SCORE_PROMPT)
    data=json.load(open('samples/sample3.json','r'))
    ob = MentalModel(data,model)
    ob.generate_MM()
    graph = ob.get_graph('abc123')
    ob.visualize_mental_model(graph)


