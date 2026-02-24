import os,cv2
import matplotlib
matplotlib.use("Agg")   # Non-GUI backend

import matplotlib.pyplot as plt
import networkx as nx
import textwrap

class MentalModel:
    def __init__(self,data,model):
        if(self.assert_valid_structure(data)):
            self.question=data['questions']
            self.answer=data['answers']
            self.concept_link=data['concept_link']
            self.concept_hierarchy=data['concept_hierarchy']
        if(model):
            self.model=model
        else:
            raise AssertionError(f"Model not initialized. Please share a valid AbstractModel object.")
    
    def assert_valid_structure(self,data):
        if not isinstance(data, dict):
            raise AssertionError("Input must be a dictionary")

        # -------------------------------------------------
        # 1. Root level keys
        # -------------------------------------------------
        required_root_keys = {
            "questions",
            "answers",
            "concept_link",
            "concept_hierarchy"
        }

        missing = required_root_keys - data.keys()
        if missing:
            raise AssertionError(f"Missing root keys: {missing}")

        # -------------------------------------------------
        # 2. Concept Hierarchy
        # -------------------------------------------------
        concept_hierarchy = data["concept_hierarchy"]
        if not isinstance(concept_hierarchy, dict):
            raise AssertionError("concept_hierarchy must be a dictionary")

        for cid, value in concept_hierarchy.items():
            if not isinstance(value, dict):
                raise AssertionError(f"concept_hierarchy[{cid}] must map to dict")

            if "name" not in value or not isinstance(value["name"], str):
                raise AssertionError(f"concept_hierarchy[{cid}] must contain string 'name'")

        concept_hierarchy_ids = set(concept_hierarchy.keys())

        # -------------------------------------------------
        # 3. Concept Link
        # -------------------------------------------------
        concept_link = data["concept_link"]
        if not isinstance(concept_link, dict):
            raise AssertionError("concept_link must be a dictionary")

        for cid, value in concept_link.items():
            if not isinstance(value, dict):
                raise AssertionError(f"concept_link[{cid}] must map to dict")

            if "name" not in value or not isinstance(value["name"], str):
                raise AssertionError(f"concept_link[{cid}] must contain string 'name'")

            if "scoring_guide" not in value or not isinstance(value["scoring_guide"], dict):
                raise AssertionError(f"concept_link[{cid}] must contain dict 'scoring_guide'")

            if "links" not in value or not isinstance(value["links"], list):
                raise AssertionError(f"concept_link[{cid}] must contain list 'links'")

            for link_id in value["links"]:
                if link_id not in concept_hierarchy_ids:
                    raise AssertionError(
                        f"concept_link[{cid}] has invalid link '{link_id}' "
                        f"(not in concept_hierarchy)"
                    )

        concept_link_ids = set(concept_link.keys())

        # -------------------------------------------------
        # 4. Questions
        # -------------------------------------------------
        questions = data["questions"]
        if not isinstance(questions, dict):
            raise AssertionError("questions must be a dictionary")

        question_ids = set(questions.keys())

        for qid, value in questions.items():
            if not isinstance(value, dict):
                raise AssertionError(f"questions[{qid}] must map to dict")

            if "T" not in value or not isinstance(value["T"], str):
                raise AssertionError(f"questions[{qid}] must contain string 'T'")

            if "I" not in value or not isinstance(value["I"], str):
                raise AssertionError(f"questions[{qid}] must contain string 'I'")

            if "CL" not in value or not isinstance(value["CL"], list):
                raise AssertionError(f"questions[{qid}] must contain list 'CL'")
            
            for cl_id in value["CL"]:
                if cl_id not in concept_link_ids:
                    raise AssertionError(
                        f"questions[{qid}] contains invalid CL id '{cl_id}' "
                        f"(not in concept_link)"
                    )

        # -------------------------------------------------
        # 5. Answers
        # -------------------------------------------------
        answers = data["answers"]
        if not isinstance(answers, dict):
            raise AssertionError("answers must be a dictionary")

        for rollno, roll_answers in answers.items():
            if not isinstance(roll_answers, dict):
                raise AssertionError(f"answers[{rollno}] must map to dict")

            for qid, ans in roll_answers.items():

                if qid not in question_ids:
                    raise AssertionError(
                        f"answers[{rollno}] contains invalid question id '{qid}'"
                    )

                if not isinstance(ans, dict):
                    raise AssertionError(
                        f"answers[{rollno}][{qid}] must map to dict"
                    )

                if "T" not in ans or not isinstance(ans["T"], str):
                    raise AssertionError(
                        f"answers[{rollno}][{qid}] must contain string 'T'"
                    )

                if "I" in ans and not isinstance(ans["I"], str):
                    raise AssertionError(
                        f"answers[{rollno}][{qid}] 'I' must be string if present"
                    )

        return True
    
    def compute_avg_concept_link(self, rollno):

        if rollno not in self.mm:
            return {}

        totals = {}
        counts = {}

        for q in self.mm[rollno]:
            for c_link, score in self.mm[rollno][q].items():
                totals[c_link] = totals.get(c_link, 0) + score
                counts[c_link] = counts.get(c_link, 0) + 1

        return {
            c: totals[c] / counts[c]
            for c in totals
        }

    # --------------------------------------------------
    # Step 2: Build Graph (Nodes + Weighted Edges)
    # --------------------------------------------------
    def build_mental_model(self, rollno):

        concept_avg = self.compute_avg_concept_link(rollno)

        graph = {
            "rollno": rollno,
            "nodes": [],
            "edges": []
        }

        # Add all hierarchy nodes
        for node_id, node_data in self.concept_hierarchy.items():
            graph["nodes"].append({
                "id": node_id,
                "label": node_data["name"]
            })

        # Add edges (concept_links)
        for c_link_id, c_link_data in self.concept_link.items():

            links = c_link_data["links"]

            # Ensure it's a valid edge (must connect 2 nodes)
            if len(links) != 2:
                continue

            node1, node2 = links

            graph["edges"].append({
                "id": c_link_id,
                "source": node1,
                "target": node2,
                "weight": concept_avg.get(c_link_id, 0)  # student mastery
            })

        return graph
    
    def generate_MM(self):
        self.mm={}
        self.mmgraph={}
        for rollno in self.answer.keys():
            self.mm[rollno]={}
            for questionno in self.answer[rollno].keys():
                data={
                        'question':{
                            "T":self.question[questionno]['T'],
                            "I":self.question[questionno]['I']
                        },
                        'answer':{
                            'T':self.answer[rollno][questionno]['T'],
                            'I':self.answer[rollno][questionno]['I'] if('I' in self.answer[rollno][questionno]) else None
                            },
                        'concept_link':"",
                        'concept_link_score':""
                }
                self.mm[rollno][questionno]={}
                for c_link in self.question[questionno]["CL"]:
                    data['concept_link']=self.concept_link[c_link]['name']
                    data['concept_link_score']=''.join(f"{k} : {v}\n" for k,v in self.concept_link[c_link]['scoring_guide'].items())
                    score=self.model.prompt_model(data)
                    self.mm[rollno][questionno][c_link]=int(score)
            temp=self.build_mental_model(rollno)
            self.mmgraph[rollno]=temp
            
    def get_graph(self,rollno):
        if(rollno in self.mmgraph):
            return self.mmgraph[rollno]
        return None
    
    def visualize_mental_model(self,graph,path='graphs'):
        if(graph==None):
            print("Empty graph object.")
        G = nx.Graph()

        # --------------------------------------------------
        # Add nodes (concept_hierarchy)
        # --------------------------------------------------
        for node in graph["nodes"]:
            G.add_node(node["id"], label=node["label"])

        # --------------------------------------------------
        # Add weighted edges (concept_link)
        # --------------------------------------------------
        for edge in graph["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"]
            )

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------
        pos = nx.spring_layout(G, k=1.5)

        plt.figure(figsize=(12, 8))

        # Draw edges
        nx.draw_networkx_edges(G, pos)

        # Draw edge weights
        edge_labels = {
            (u, v): round(d["weight"], 2)
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Draw rectangular node labels using stored label
        for node_id, (x, y) in pos.items():
            label = G.nodes[node_id]["label"]
            wrapped = "\n".join(textwrap.wrap(label, 30))

            plt.text(
                x, y, wrapped,
                ha='center',
                va='center',
                bbox=dict(boxstyle="square", pad=0.6)
            )

        plt.title(f"Mental Model of student {graph['rollno']}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{path}/mental_model_{graph['rollno']}.png", dpi=300, bbox_inches="tight")
        plt.close()