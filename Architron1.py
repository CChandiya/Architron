import streamlit as st
import spacy
import graphviz
import random
import pandas as pd
from collections import defaultdict
import json
import os
import requests
from dotenv import load_dotenv
import ast
import re
import networkx as nx
import matplotlib.pyplot as plt
from groq import Groq
import io
import base64
from typing import Dict, List, Set, Tuple

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
code_analysis_client = Groq(api_key=GROQ_API_KEY) 
# Python Parser using AST
client = Groq(api_key="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")

# Language Parsers
#------------------------------------------------------------------

# Python Parser using AST
class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.current_class = None
    
    def visit_ClassDef(self, node):
        class_info = {
            'name': node.name,
            'methods': [],
            'attributes': [],
            'parents': []
        }
        
        # Extract parent classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info['parents'].append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info['parents'].append(base.attr)
        
        old_class = self.current_class
        self.current_class = class_info
        
        # Visit all nodes in the class body
        for item in node.body:
            self.visit(item)
        
        self.classes.append(class_info)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        if self.current_class:
            self.current_class['methods'].append(node.name)
        else:
            self.functions.append(node.name)
    
    def visit_Assign(self, node):
        if self.current_class:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.current_class['attributes'].append(target.id)
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)
    
    def visit_ImportFrom(self, node):
        if node.module:
            for name in node.names:
                self.imports.append(f"{node.module}.{name.name}")

def extract_python_components(code: str) -> Dict:
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Create a visitor and visit all nodes
        visitor = CodeVisitor()
        visitor.visit(tree)
        
        return {
            'classes': visitor.classes,
            'functions': visitor.functions,
            'imports': visitor.imports
        }
    except SyntaxError as e:
        st.error(f"Syntax error in Python code: {str(e)}")
        return {'classes': [], 'functions': [], 'imports': []}
    except Exception as e:
        st.error(f"Error parsing Python code: {str(e)}")
        return {'classes': [], 'functions': [], 'imports': []}

# JavaScript Parser using regex patterns
def extract_javascript_components(code: str) -> Dict:
    classes = []
    functions = []
    imports = []
    
    # Match class definitions with inheritance
    class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{([^}]*)\}'
    class_matches = re.finditer(class_pattern, code)
    
    for match in class_matches:
        class_name = match.group(1)
        parent_class = match.group(2)
        class_body = match.group(3)
        
        class_info = {
            'name': class_name,
            'methods': [],
            'attributes': [],
            'parents': [parent_class] if parent_class else []
        }
        
        # Extract methods
        method_pattern = r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{[^}]*\}'
        method_matches = re.finditer(method_pattern, class_body)
        for method_match in method_matches:
            class_info['methods'].append(method_match.group(1))
        
        # Extract properties/attributes (simplified)
        attribute_pattern = r'(?:this\.)?(\w+)\s*='
        attribute_matches = re.finditer(attribute_pattern, class_body)
        for attr_match in attribute_matches:
            attr_name = attr_match.group(1)
            if attr_name not in class_info['methods']:
                class_info['attributes'].append(attr_name)
        
        classes.append(class_info)
    
    # Match standalone functions
    function_pattern = r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:\([^)]*\)|async\s+\([^)]*\))\s*=?>?\s*\{'
    function_matches = re.finditer(function_pattern, code)
    for match in function_matches:
        functions.append(match.group(1))
    
    # Match imports
    import_pattern = r'import\s+(?:{[^}]*}|[^;]*)\s+from\s+[\'"]([^\'"]*)[\'"]'
    import_matches = re.finditer(import_pattern, code)
    for match in import_matches:
        imports.append(match.group(1))
    
    return {
        'classes': classes,
        'functions': functions,
        'imports': imports
    }

# Java Parser using regex patterns
def extract_java_components(code: str) -> Dict:
    classes = []
    functions = []  # In Java, these are standalone methods
    imports = []
    
    # Match imports
    import_pattern = r'import\s+([^;]+);'
    import_matches = re.finditer(import_pattern, code)
    for match in import_matches:
        imports.append(match.group(1))
    
    # Match class definitions with inheritance and interfaces
    class_pattern = r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{([^}]*(?:\{[^}]*\})*[^}]*)\}'
    class_matches = re.finditer(class_pattern, code)
    
    for match in class_matches:
        class_name = match.group(1)
        parent_class = match.group(2)
        interfaces = match.group(3)
        class_body = match.group(4)
        
        class_info = {
            'name': class_name,
            'methods': [],
            'attributes': [],
            'parents': []
        }
        
        # Add parent class if exists
        if parent_class:
            class_info['parents'].append(parent_class)
        
        # Add interfaces if they exist
        if interfaces:
            for interface in interfaces.split(','):
                class_info['parents'].append(interface.strip())
        
        # Extract methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static|final|abstract)?\s*(?:<[^>]*>)?\s*\w+\s+(\w+)\s*\([^\)]*\)'
        method_matches = re.finditer(method_pattern, class_body)
        for method_match in method_matches:
            class_info['methods'].append(method_match.group(1))
        
        # Extract attributes
        attribute_pattern = r'(?:public|private|protected)?\s*(?:static|final)?\s*\w+\s+(\w+)\s*(?:=|;)'
        attribute_matches = re.finditer(attribute_pattern, class_body)
        for attr_match in attribute_matches:
            class_info['attributes'].append(attr_match.group(1))
        
        classes.append(class_info)
    
    # Find standalone methods (unlikely in Java but possible in certain contexts)
    standalone_method_pattern = r'(?:public|private|protected)?\s*(?:static)\s*\w+\s+(\w+)\s*\([^\)]*\)'
    standalone_matches = re.finditer(standalone_method_pattern, code)
    for match in standalone_matches:
        method_name = match.group(1)
        # Check if it's not already in a class
        is_in_class = False
        for class_info in classes:
            if method_name in class_info['methods']:
                is_in_class = True
                break
        if not is_in_class:
            functions.append(method_name)
    
    return {
        'classes': classes,
        'functions': functions,
        'imports': imports
    }

# C# Parser using regex patterns
def extract_csharp_components(code: str) -> Dict:
    classes = []
    functions = []
    imports = []
    
    # Match using statements (imports)
    import_pattern = r'using\s+([^;]+);'
    import_matches = re.finditer(import_pattern, code)
    for match in import_matches:
        imports.append(match.group(1))
    
    # Match class definitions with inheritance
    class_pattern = r'(?:public|private|protected|internal)?\s*(?:static|abstract|sealed)?\s*class\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{([^}]*(?:\{[^}]*\})*[^}]*)\}'
    class_matches = re.finditer(class_pattern, code)
    
    for match in class_matches:
        class_name = match.group(1)
        inheritance = match.group(2)
        class_body = match.group(3)
        
        class_info = {
            'name': class_name,
            'methods': [],
            'attributes': [],
            'parents': []
        }
        
        # Add parent classes and interfaces
        if inheritance:
            for parent in inheritance.split(','):
                class_info['parents'].append(parent.strip())
        
        # Extract methods
        method_pattern = r'(?:public|private|protected|internal)?\s*(?:static|virtual|abstract|override|async)?\s*(?:[<][^>]*[>])?\s*\w+\s+(\w+)\s*\([^\)]*\)'
        method_matches = re.finditer(method_pattern, class_body)
        for method_match in method_matches:
            class_info['methods'].append(method_match.group(1))
        
        # Extract properties and fields
        property_pattern = r'(?:public|private|protected|internal)?\s*(?:static|readonly)?\s*\w+\s+(\w+)\s*(?:\{|=|;)'
        property_matches = re.finditer(property_pattern, class_body)
        for prop_match in property_matches:
            prop_name = prop_match.group(1)
            if prop_name not in class_info['methods']:
                class_info['attributes'].append(prop_name)
        
        classes.append(class_info)
    
    # Find standalone methods (typically in static classes)
    standalone_method_pattern = r'(?:public|private|protected|internal)?\s*static\s*\w+\s+(\w+)\s*\([^\)]*\)'
    standalone_matches = re.finditer(standalone_method_pattern, code)
    for match in standalone_matches:
        method_name = match.group(1)
        # Check if it's not already in a class
        is_in_class = False
        for class_info in classes:
            if method_name in class_info['methods']:
                is_in_class = True
                break
        if not is_in_class:
            functions.append(method_name)
    
    return {
        'classes': classes,
        'functions': functions,
        'imports': imports
    }

# Language-agnostic LLM parser
def extract_language_agnostic_llm(code: str, language: str) -> Dict:
    prompt = f"""
    Extract the code architecture from the following {language} code. Identify:
    1. Classes (with their methods and attributes)
    2. Functions
    3. Imports/dependencies
    4. Relationships between components

    Format your response as a JSON object with these keys:
    - classes: array of class objects with properties: name, methods, attributes, parents
    - functions: array of function names
    - imports: array of import strings
    - relationships: array of relationship objects with properties: from, to, type (inheritance, composition, dependency, etc.)

    Code:
    ```
    {code}
    ```
    
    Return ONLY the JSON object without any explanation. The JSON must be valid and parsable.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON if present
        json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                extracted_json = json_match.group(1).strip()
                return json.loads(extracted_json)
            except json.JSONDecodeError:
                # Try to extract just a clean JSON object without any markdown formatting
                json_obj_match = re.search(r'(\{[\s\S]*\})', response_text, re.DOTALL)
                if json_obj_match:
                    try:
                        return json.loads(json_obj_match.group(1))
                    except:
                        pass
        
        # If no JSON found or parsing failed, return an empty structure
        st.warning("LLM returned a response but it couldn't be parsed as JSON. Using minimal structure.")
        return {
            "classes": [],
            "functions": [],
            "imports": [],
            "relationships": [],
            "raw_analysis": response_text
        }
    except Exception as e:
        st.error(f"Error with LLM analysis: {str(e)}")
        return {
            "classes": [],
            "functions": [],
            "imports": [],
            "relationships": [],
            "error": str(e)
        }

# Function to detect language based on code patterns
def detect_language(code: str) -> str:
    # Check for Python syntax
    if re.search(r'def\s+\w+\s*\(.*\):', code) or re.search(r'import\s+\w+', code):
        return "Python"
    
    # Check for JavaScript/TypeScript syntax
    if re.search(r'const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|function\s+\w+\s*\(', code) or re.search(r'import\s+.*\s+from\s+', code):
        return "JavaScript"
    
    # Check for Java syntax
    if re.search(r'public\s+class\s+\w+|import\s+java\.', code):
        return "Java"
    
    # Check for C# syntax
    if re.search(r'using\s+System;|namespace\s+\w+|public\s+class\s+\w+', code):
        return "C#"
    
    # Check for TypeScript-specific syntax
    if re.search(r'interface\s+\w+\s*\{|type\s+\w+\s*=', code):
        return "TypeScript"
    
    # Check for PHP syntax
    if re.search(r'\<\?php|\$\w+\s*=', code):
        return "PHP"
    
    # Check for Ruby syntax
    if re.search(r'def\s+\w+\s*(\|.*\|)?|require\s+[\'\"]', code):
        return "Ruby"
    
    # Check for Go syntax
    if re.search(r'package\s+\w+|func\s+\w+\s*\(|import\s+\(', code):
        return "Go"
    
    # Default to unknown
    return "Unknown"

# Function to get components based on language
def extract_components_by_language(code: str, language: str, use_llm: bool = False) -> Dict:
    if use_llm:
        return extract_language_agnostic_llm(code, language)
    
    if language == "Python":
        return extract_python_components(code)
    elif language in ["JavaScript", "TypeScript"]:
        return extract_javascript_components(code)
    elif language == "Java":
        return extract_java_components(code)
    elif language == "C#":
        return extract_csharp_components(code)
    else:
        # For unsupported languages, fall back to LLM analysis
        st.warning(f"Native parser not available for {language}. Using LLM analysis.")
        return extract_language_agnostic_llm(code, language)

# Function to use LLM to analyze code relationships
def analyze_code_with_llm(code: str, language: str) -> Dict:
    prompt = f"""
    Analyze the following {language} code and describe the architecture, relationships between components,
    and how data flows through the system. Specifically identify:
    
    1. Main components (classes, modules, functions)
    2. Relationships between components (inheritance, composition, dependencies)
    3. Data flow
    4. Function calls and interactions
    
    Format your response as a structured JSON with these components.
    
    Code:
    ```
    {code}
    ```
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON if present
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            import json
            try:
                return json.loads(json_match.group(1))
            except:
                pass
                
        # If no JSON found or parsing failed, return the raw response
        return {"raw_analysis": response_text}
    except Exception as e:
        st.error(f"Error with LLM analysis: {str(e)}")
        return {"error": str(e)}

# Function to generate a Mermaid diagram from code analysis
def generate_mermaid_diagram(analysis: Dict, language: str) -> str:
    mermaid_code = "classDiagram\n"
    
    # Add classes from analysis
    if 'classes' in analysis:
        for cls in analysis['classes']:
            mermaid_code += f"    class {cls['name']} {{\n"
            
            for attr in cls.get('attributes', []):
                mermaid_code += f"        +{attr}\n"
            
            for method in cls.get('methods', []):
                mermaid_code += f"        +{method}()\n"
            
            mermaid_code += "    }\n"
    
    # Add standalone functions if needed
    if 'functions' in analysis and analysis['functions']:
        if language == "Python":
            mermaid_code += "    class Functions {\n"
        else:
            mermaid_code += f"    class {language}Functions {{\n"
            
        for func in analysis['functions']:
            mermaid_code += f"        +{func}()\n"
        mermaid_code += "    }\n"
    
    # Add relationships
    if 'relationships' in analysis:
        for rel in analysis.get('relationships', []):
            if isinstance(rel, dict) and 'from' in rel and 'to' in rel and 'type' in rel:
                relation_type = rel['type']
                if relation_type == 'inheritance':
                    mermaid_code += f"    {rel['from']} --|> {rel['to']}\n"
                elif relation_type == 'composition':
                    mermaid_code += f"    {rel['from']} *-- {rel['to']}\n"
                elif relation_type == 'aggregation':
                    mermaid_code += f"    {rel['from']} o-- {rel['to']}\n"
                elif relation_type == 'dependency':
                    mermaid_code += f"    {rel['from']} ..> {rel['to']}\n"
                else:
                    mermaid_code += f"    {rel['from']} -- {rel['to']}: {relation_type}\n"
    
    # Add inheritance relationships from classes
    for cls in analysis.get('classes', []):
        for parent in cls.get('parents', []):
            mermaid_code += f"    {cls['name']} --|> {parent}\n"
    
    # Add import relationships
    if 'imports' in analysis and analysis['imports']:
        for i, imp in enumerate(analysis['imports'][:10]): 
            import_name = imp.split('.')[-1] if '.' in imp else imp
            if import_name not in [cls['name'] for cls in analysis.get('classes', [])]:
                mermaid_code += f"    class {import_name} <<external>> {{\n    }}\n"
                
                # Add dependencies to imported modules
                for cls in analysis.get('classes', []):
                    mermaid_code += f"    {cls['name']} ..> {import_name}: imports\n"
    
    return mermaid_code

# Function to generate NetworkX diagram
def generate_networkx_diagram(analysis: Dict, language: str) -> plt.Figure:
    G = nx.DiGraph()
    
    # Add nodes for classes
    for cls in analysis.get('classes', []):
        G.add_node(cls['name'], type='class')
        
        # Add methods as nodes and connect them to their class
        for method in cls.get('methods', []):
            method_node = f"{cls['name']}.{method}"
            G.add_node(method_node, type='method')
            G.add_edge(cls['name'], method_node, type='has_method')
    
    # Add nodes for standalone functions
    for func in analysis.get('functions', []):
        G.add_node(func, type='function')
    
    # Add inheritance relationships
    for cls in analysis.get('classes', []):
        for parent in cls.get('parents', []):
            G.add_node(parent, type='external_class')
            G.add_edge(cls['name'], parent, type='inherits')
    
    # Add relationships if they exist
    if 'relationships' in analysis:
        for rel in analysis.get('relationships', []):
            if isinstance(rel, dict) and 'from' in rel and 'to' in rel:
                source = rel['from']
                target = rel['to']
                
                # Add nodes if they don't exist
                if source not in G.nodes():
                    G.add_node(source, type='component')
                if target not in G.nodes():
                    G.add_node(target, type='component')
                
                G.add_edge(source, target, type=rel.get('type', 'unknown'))
    
    # Add imports as nodes
    for i, imp in enumerate(analysis.get('imports', [])[:10]):  # Limit to avoid clutter
        import_name = imp.split('.')[-1] if '.' in imp else imp
        if import_name not in G.nodes():
            G.add_node(import_name, type='import')
            
            # Connect imports to classes that might use them
            for cls in analysis.get('classes', []):
                G.add_edge(cls['name'], import_name, type='uses')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, f"No components found to display for {language} code", 
                horizontalalignment='center', fontsize=14)
        plt.axis('off')
        return fig
    
    # Generate layout
    pos = nx.spring_layout(G, k=0.9, seed=42)
    
    # Draw nodes with different colors based on type
    class_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'class']
    external_class_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'external_class']
    method_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'method']
    function_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'function']
    component_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'component']
    import_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'import']
    
    node_size_base = 1500
    
    if class_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_color='lightblue', 
                              node_size=node_size_base, alpha=0.8, node_shape='s')
    if external_class_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=external_class_nodes, node_color='lightgray', 
                              node_size=node_size_base - 200, alpha=0.6, node_shape='s')
    if method_nodes:                          
        nx.draw_networkx_nodes(G, pos, nodelist=method_nodes, node_color='lightgreen', 
                              node_size=node_size_base - 500, alpha=0.6)
    if function_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=function_nodes, node_color='salmon', 
                              node_size=node_size_base - 300, alpha=0.7)
    if component_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=component_nodes, node_color='orange', 
                              node_size=node_size_base - 300, alpha=0.7, node_shape='h')
    if import_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=import_nodes, node_color='purple', 
                              node_size=node_size_base - 600, alpha=0.5, node_shape='d')
    
    # Draw edges with different styles based on relationship type
    inheritance_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'inherits']
    method_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'has_method']
    uses_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'uses']
    other_edges = [(u, v) for u, v, d in G.edges(data=True) 
                  if d.get('type') not in ['inherits', 'has_method', 'uses']]
    
    if inheritance_edges:
        nx.draw_networkx_edges(G, pos, edgelist=inheritance_edges, edge_color='blue', 
                              arrows=True, arrowstyle='->', arrowsize=20, width=2)
    if method_edges:
        nx.draw_networkx_edges(G, pos, edgelist=method_edges, edge_color='green', 
                              arrows=True, style='dashed', width=1)
    if uses_edges:
        nx.draw_networkx_edges(G, pos, edgelist=uses_edges, edge_color='purple', 
                              arrows=True, style='dotted', width=1)
    if other_edges:
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='red', 
                              arrows=True, width=1.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add a legend
    legend_elements = []
    if class_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue',
                              markersize=15, label='Class'))
    if external_class_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
                              markersize=15, label='External Class'))
    if method_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                              markersize=15, label='Method'))
    if function_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon',
                              markersize=15, label='Function'))
    if component_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='orange',
                              markersize=15, label='Component'))
    if import_nodes:
        legend_elements.append(plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='purple',
                              markersize=15, label='Import/Module'))
    if inheritance_edges:
        legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Inheritance'))
    if method_edges:
        legend_elements.append(plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='Has Method'))
    if uses_edges:
        legend_elements.append(plt.Line2D([0], [0], color='purple', lw=2, linestyle=':', label='Uses/Imports'))
    if other_edges:
        legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Other Relationship'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"{language} Code Architecture", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    return fig

# Load NLP model
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# Define software categories with expanded descriptions
architecture_types = {
    "web application": {
        "name": "MVC (Model-View-Controller)",
        "description": "Separates application logic into three interconnected components: Model (data), View (UI), and Controller (business logic).",
        "best_for": "Web applications with clear separation of concerns."
    },
    "real-time system": {
        "name": "Event-Driven Architecture",
        "description": "Design pattern built around the production, detection, and consumption of events.",
        "best_for": "Real-time applications, IoT, trading platforms, chat applications."
    },
    "scalable system": {
        "name": "Microservices Architecture",
        "description": "Structures an application as a collection of loosely coupled services.",
        "best_for": "Large-scale applications requiring independent scaling and deployment."
    },
    "data processing": {
        "name": "Batch Processing Architecture",
        "description": "Processes data in large groups at scheduled intervals rather than in real-time.",
        "best_for": "Data analytics, ETL workflows, reporting systems."
    },
    "ai model": {
        "name": "Serverless Architecture",
        "description": "Builds applications that don't require server management, using FaaS (Function as a Service).",
        "best_for": "ML model deployment, intermittent workloads, rapid prototyping."
    },
    "mobile": {
        "name": "Client-Server Architecture",
        "description": "Distributes application components between a client and a server.",
        "best_for": "Mobile apps, desktop applications with centralized data."
    },
    "distributed": {
        "name": "Peer-to-Peer Architecture",
        "description": "Distributes workloads across equally privileged participants.",
        "best_for": "File sharing, blockchain applications, distributed computing."
    }
}

# Expanded technology stacks with pros and cons, now organized by programming language
tech_stacks_by_language = {
    "Python": {
        "MVC (Model-View-Controller)": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Django", "Flask", "FastAPI"],
            "database": ["PostgreSQL", "MySQL", "MongoDB", "SQLite"],
            "pros": ["Python's readability makes MVC components clear", "Django's built-in admin interface", "Rapid development with Python frameworks"],
            "cons": ["Performance can be slower than compiled languages", "GIL limitations for CPU-bound tasks"]
        },
        "Event-Driven Architecture": {
            "frontend": ["React", "Vue.js", "Socket.IO"],
            "backend": ["FastAPI with WebSockets", "Django Channels", "Tornado"],
            "database": ["MongoDB", "Redis", "Cassandra"],
            "pros": ["Python's async features work well with events", "Excellent library support for messaging", "Good for prototyping event systems"],
            "cons": ["Async complexity in Python", "Performance under high loads"]
        },
        "Microservices Architecture": {
            "frontend": ["React", "Vue.js", "Next.js"],
            "backend": ["FastAPI", "Flask", "Django REST Framework"],
            "infrastructure": ["Docker", "Kubernetes", "AWS ECS"],
            "database": ["PostgreSQL", "MongoDB", "Redis"],
            "pros": ["Easy service isolation with Python modules", "Great API tools", "Simple deployments with Docker"],
            "cons": ["Communication overhead", "Service coordination complexity"]
        },
        "Serverless Architecture": {
            "functions": ["AWS Lambda with Python", "Azure Functions with Python", "Google Cloud Functions with Python"],
            "api": ["API Gateway with Lambda", "FastAPI", "Flask"],
            "database": ["DynamoDB", "Firestore", "MongoDB Atlas"],
            "frontend": ["React", "Vue.js", "Next.js"],
            "pros": ["Python's lightweight nature suits serverless", "Great library ecosystem", "Rapid development"],
            "cons": ["Cold starts with Python runtimes", "Package dependency challenges"]
        }
    },
    "JavaScript": {
        "MVC (Model-View-Controller)": {
            "frontend": ["React", "Vue.js", "Angular", "Svelte"],
            "backend": ["Express.js", "NestJS", "Koa"],
            "database": ["MongoDB", "PostgreSQL", "MySQL"],
            "pros": ["Same language for frontend and backend", "Vast npm ecosystem", "Non-blocking I/O"],
            "cons": ["Callback hell if not properly structured", "Prototype inheritance complexity"]
        },
        "Event-Driven Architecture": {
            "frontend": ["React", "Vue.js", "Socket.IO"],
            "backend": ["Node.js with Socket.IO", "NestJS", "Express with EventEmitter"],
            "database": ["MongoDB", "Redis", "RethinkDB"],
            "pros": ["JavaScript's async nature fits event models perfectly", "WebSockets support", "Event loop for concurrency"],
            "cons": ["Single-threaded bottlenecks", "Memory leaks with improper event handling"]
        },
        "Microservices Architecture": {
            "frontend": ["React", "Angular", "Next.js", "Vue.js"],
            "backend": ["Express.js", "NestJS", "Fastify"],
            "infrastructure": ["Docker", "Kubernetes", "AWS ECS"],
            "database": ["MongoDB", "PostgreSQL", "Redis"],
            "pros": ["Small, focused services", "Independent deployments", "Easier scaling"],
            "cons": ["Service discovery complexity", "Transaction management across services"]
        },
        "Serverless Architecture": {
            "functions": ["AWS Lambda with Node.js", "Vercel Serverless", "Netlify Functions"],
            "api": ["API Gateway", "Express.js", "NestJS"],
            "database": ["DynamoDB", "Firestore", "MongoDB Atlas", "FaunaDB"],
            "frontend": ["React", "Next.js", "Vue.js", "Svelte"],
            "pros": ["JavaScript's event loop works great in serverless", "Small bundle sizes", "Rapid startup"],
            "cons": ["Cold starts", "Limited execution duration"]
        }
    },
    "Java": {
        "MVC (Model-View-Controller)": {
            "frontend": ["React", "Angular", "Vue.js"],
            "backend": ["Spring MVC", "Spring Boot", "Jakarta EE"],
            "database": ["PostgreSQL", "MySQL", "Oracle", "SQL Server"],
            "pros": ["Strong typing and robust architecture", "Enterprise-grade frameworks", "Mature ecosystem"],
            "cons": ["Verbose coding style", "Steeper learning curve", "Larger deployments"]
        },
        "Event-Driven Architecture": {
            "frontend": ["React", "Angular", "Vue.js"],
            "backend": ["Spring WebFlux", "Vert.x", "Akka"],
            "database": ["Cassandra", "Kafka", "Redis", "Elasticsearch"],
            "pros": ["Excellent concurrency model", "Robust event-processing", "Enterprise reliability"],
            "cons": ["Complex setup", "Steeper learning curve"]
        },
        "Microservices Architecture": {
            "frontend": ["React", "Angular", "Vue.js"],
            "backend": ["Spring Boot", "Quarkus", "Micronaut"],
            "infrastructure": ["Docker", "Kubernetes", "Istio"],
            "database": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"],
            "pros": ["Strong typing across services", "Rich ecosystem", "Enterprise support"],
            "cons": ["Resource intensive", "Complex deployments"]
        },
        "Serverless Architecture": {
            "functions": ["AWS Lambda with Java", "Azure Functions with Java", "Google Cloud Functions with Java"],
            "api": ["API Gateway", "Spring Cloud Function", "Micronaut"],
            "database": ["DynamoDB", "Firestore", "MongoDB Atlas", "Aurora Serverless"],
            "frontend": ["React", "Angular", "Vue.js"],
            "pros": ["Strong type safety", "Mature tooling", "Enterprise-grade reliability"],
            "cons": ["Cold start latency", "Larger deployment packages", "Higher memory consumption"]
        }
    },
    "TypeScript": {
        "MVC (Model-View-Controller)": {
            "frontend": ["React with TypeScript", "Angular", "Vue.js with TypeScript"],
            "backend": ["NestJS", "Express with TypeScript", "Koa with TypeScript"],
            "database": ["PostgreSQL", "MongoDB", "MySQL"],
            "pros": ["Type safety across frontend and backend", "Better IDE support", "Fewer runtime errors"],
            "cons": ["Additional compilation step", "Learning curve for type system"]
        },
        "Event-Driven Architecture": {
            "frontend": ["React with TypeScript", "Angular", "Vue.js with TypeScript"],
            "backend": ["NestJS", "Node.js with TypeScript", "TypeORM"],
            "database": ["MongoDB", "Redis", "PostgreSQL"],
            "pros": ["Type-safe events and handlers", "Better maintainability", "IDE autocompletion"],
            "cons": ["Additional setup complexity", "Type definition management"]
        },
        "Microservices Architecture": {
            "frontend": ["React with TypeScript", "Angular", "Next.js with TypeScript"],
            "backend": ["NestJS", "Express with TypeScript", "Fastify with TypeScript"],
            "infrastructure": ["Docker", "Kubernetes", "AWS ECS"],
            "database": ["PostgreSQL", "MongoDB", "Redis"],
            "pros": ["Type consistency across services", "Contract enforcements", "Better refactoring"],
            "cons": ["More boilerplate code", "Type definition sharing between services"]
        },
        "Serverless Architecture": {
            "functions": ["AWS Lambda with TypeScript", "Vercel Serverless", "Netlify Functions with TypeScript"],
            "api": ["API Gateway", "NestJS", "Express with TypeScript"],
            "database": ["DynamoDB", "Firestore", "MongoDB Atlas", "FaunaDB"],
            "frontend": ["React with TypeScript", "Next.js", "Angular"],
            "pros": ["Type safety in distributed functions", "Better IDE support", "Reduced runtime errors"],
            "cons": ["Compilation overhead", "Type definition management across functions"]
        }
    },
    "Go": {
        "MVC (Model-View-Controller)": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Gin", "Echo", "Revel"],
            "database": ["PostgreSQL", "MySQL", "MongoDB"],
            "pros": ["High performance", "Statically typed", "Small memory footprint"],
            "cons": ["Less mature web frameworks", "Frontend requires different language"]
        },
        "Event-Driven Architecture": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Go with channels", "NATS", "RabbitMQ client"],
            "database": ["Redis", "Cassandra", "CockroachDB"],
            "pros": ["Goroutines for concurrent event handling", "Built-in channels", "Low latency"],
            "cons": ["Less ecosystem maturity", "Frontend in different language"]
        },
        "Microservices Architecture": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Go Micro", "Gin", "Go Kit"],
            "infrastructure": ["Docker", "Kubernetes", "Istio"],
            "database": ["PostgreSQL", "MongoDB", "Redis"],
            "pros": ["Small binaries for microservices", "Low resource usage", "Fast startup"],
            "cons": ["Fewer frameworks compared to other languages", "Less code reuse"]
        },
        "Serverless Architecture": {
            "functions": ["AWS Lambda with Go", "Google Cloud Functions with Go", "OpenFaaS with Go"],
            "api": ["API Gateway", "Gin", "Echo"],
            "database": ["DynamoDB", "Firestore", "MongoDB Atlas"],
            "frontend": ["React", "Vue.js", "Angular"],
            "pros": ["Fast cold starts", "Small deployment size", "Memory efficiency"],
            "cons": ["Less feature-rich ecosystem", "Frontend in different language"]
        }
    }
}

# Get the traditional tech stacks format from the language-specific ones for backward compatibility
tech_stacks = {
    arch_name: stack
    for language in tech_stacks_by_language.values()
    for arch_name, stack in language.items()
}

# Project requirement templates
project_templates = {
    "E-commerce": "Build a scalable web application for selling products online with user accounts, product catalog, shopping cart, and payment processing.",
    "Social Media": "Create a real-time platform for users to share posts, connect with friends, and receive notifications about activities.",
    "Data Analytics": "Develop a system to process large datasets, generate insights, and visualize trends with interactive dashboards.",
    "Mobile App": "Build a mobile application with a backend API to manage user data, notifications, and content delivery.",
    "IoT Platform": "Create a real-time system to collect, process, and analyze data from IoT devices with alerts and dashboards.",
    "SaaS Application": "Develop a subscription-based software service with user management, payment processing, and multiple service tiers.",
    "AI/ML Platform": "Build a platform that allows users to train, deploy, and monitor machine learning models."
}

# Define deployment cost estimator
def estimate_deployment_cost(architecture, scale="medium"):
    cost_estimates = {
        "MVC (Model-View-Controller)": {"small": "$50-100/mo", "medium": "$150-300/mo", "large": "$500-1000/mo"},
        "Event-Driven Architecture": {"small": "$100-200/mo", "medium": "$300-600/mo", "large": "$1000-2000/mo"},
        "Microservices Architecture": {"small": "$200-400/mo", "medium": "$500-1000/mo", "large": "$2000-5000/mo"},
        "Batch Processing Architecture": {"small": "$150-300/mo", "medium": "$400-800/mo", "large": "$1500-3000/mo"},
        "Serverless Architecture": {"small": "$20-100/mo", "medium": "$100-400/mo", "large": "$500-1500/mo"},
        "Client-Server Architecture": {"small": "$50-150/mo", "medium": "$200-500/mo", "large": "$600-1500/mo"},
        "Peer-to-Peer Architecture": {"small": "$30-100/mo", "medium": "$150-400/mo", "large": "$500-1200/mo"},
    }
    
    return cost_estimates.get(architecture, {}).get(scale, "N/A")

# Define function to generate project roadmap
def generate_roadmap(architecture_type):
    phases = [
        {"name": "Planning & Requirements", "duration": "2-4 weeks", "activities": [
            "Gather requirements", 
            "Define user stories", 
            "Create technical specifications",
            "Choose technology stack"
        ]},
        {"name": "Design", "duration": "3-5 weeks", "activities": [
            "Create architecture diagrams",
            "Design database schema",
            "Define API endpoints",
            "Create UI/UX mockups"
        ]},
        {"name": "Development", "duration": "8-16 weeks", "activities": [
            "Set up development environment",
            "Implement core features",
            "Develop APIs",
            "Create frontend components",
            "Implement database models"
        ]},
        {"name": "Testing", "duration": "4-6 weeks", "activities": [
            "Unit testing",
            "Integration testing",
            "Performance testing",
            "User acceptance testing"
        ]},
        {"name": "Deployment", "duration": "2-3 weeks", "activities": [
            "Set up production environment",
            "Configure CI/CD pipeline",
            "Deploy application",
            "Monitor performance"
        ]},
        {"name": "Maintenance", "duration": "Ongoing", "activities": [
            "Bug fixes",
            "Feature enhancements",
            "Performance optimization",
            "Security updates"
        ]}
    ]
    
    # Add architecture-specific activities
    if "Microservices" in architecture_type:
        phases[2]["activities"].extend(["Set up service discovery", "Implement API gateway"])
    elif "Event-Driven" in architecture_type:
        phases[2]["activities"].extend(["Configure message brokers", "Implement event handlers"])
    elif "Serverless" in architecture_type:
        phases[2]["activities"].extend(["Configure cloud functions", "Set up API gateway"])
        
    return phases

# Function to generate code using Groq API
# Function to generate code using Groq API
def generate_code_with_groq(project_description, architecture, frontend, backend, database, language="Python", component="all"):
    if not GROQ_API_KEY:
        return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Define specific coding guidelines to improve code quality
    coding_guidelines = """
    Your code should:
    1. Follow production-ready best practices for the chosen languages and frameworks
    2. Include proper error handling with try/catch blocks
    3. Include comments explaining complex logic
    4. Use consistent naming conventions
    5. Implement proper authentication and authorization if applicable
    6. Follow security best practices (input validation, parameterized queries, etc.)
    7. Be well-structured with clear separation of concerns
    8. Include tests or test placeholders where appropriate
    9. Use environment variables for configuration
    10. Include all necessary dependencies and setup instructions
    11. Focus on functionality over comments - the code should actually work when copied
    
    Do not include placeholder comments like "// Add more code here". Provide complete, runnable code.
    """
    
    # Create language-specific instructions
    language_specific = ""
    if language == "Python":
        language_specific = """
        For Python code:
        - Use Python 3.9+ syntax
        - Follow PEP 8 style guidelines
        - Use virtual environments and requirements.txt
        - Include docstrings for functions and classes
        - Use type hints where appropriate
        """
    elif language == "JavaScript" or language == "TypeScript":
        language_specific = """
        For JavaScript/TypeScript code:
        - Use ES6+ syntax
        - Properly set up package.json with all dependencies
        - Use async/await for asynchronous operations
        - Include proper TypeScript types if using TypeScript
        - Prefer functional components in React
        """
    elif language == "Java":
        language_specific = """
        For Java code:
        - Use Java 11+ syntax
        - Include Maven or Gradle build files
        - Follow standard Java naming conventions
        - Use appropriate design patterns
        - Include proper exception handling
        """
    elif language == "Go":
        language_specific = """
        For Go code:
        - Follow Go idioms and standard library patterns
        - Include proper error handling
        - Use go modules for dependency management
        - Follow standard Go naming conventions
        - Include appropriate tests
        """
    
    # Framework-specific instructions
    framework_specific = ""
    if backend == "Django" or backend == "Django REST Framework":
        framework_specific = """
        For Django:
        - Include proper URL routing
        - Use Django ORM models with proper relationships
        - Implement Django views or viewsets correctly
        - Include serializers if using DRF
        - Set up proper settings.py configuration
        """
    elif backend == "Flask" or backend == "FastAPI":
        framework_specific = """
        For Flask/FastAPI:
        - Set up proper routing and middleware
        - Implement proper request validation
        - Use SQLAlchemy if needed for database models
        - Include proper error handling and status codes
        - Implement dependency injection (for FastAPI)
        """
    elif backend == "Express.js" or backend == "NestJS":
        framework_specific = """
        For Express/NestJS:
        - Implement proper middleware and routing
        - Set up controller and service layers
        - Use appropriate ORM (Mongoose, TypeORM, Prisma)
        - Implement proper validation
        - Include error handling middleware
        """
    elif backend == "Spring Boot":
        framework_specific = """
        For Spring Boot:
        - Use appropriate annotations
        - Implement controller, service, and repository layers
        - Use proper dependency injection
        - Set up proper application.properties/yml
        - Include exception handling
        """
    
    # Database-specific instructions
    database_specific = ""
    if database == "PostgreSQL" or database == "MySQL":
        database_specific = """
        For SQL databases:
        - Use proper schema design with appropriate relationships
        - Include indexes for frequent queries
        - Use migrations for schema changes
        - Implement transactions where needed
        - Use parameterized queries to prevent SQL injection
        """
    elif database == "MongoDB":
        database_specific = """
        For MongoDB:
        - Design schemas with proper document structure
        - Use appropriate indexes
        - Implement proper data validation
        - Consider embedding vs referencing based on access patterns
        - Use aggregation pipeline for complex queries
        """
    elif database == "Redis":
        database_specific = """
        For Redis:
        - Use appropriate data structures (strings, hashes, lists, sets)
        - Implement proper expiration policies
        - Consider caching strategies
        - Use transactions where appropriate
        - Handle connection pooling properly
        """
    
    # Customize the prompt based on which component to generate
    if component == "frontend":
        prompt = f"""
Generate complete, production-ready frontend code for a {architecture} application with the following specification:

Project Description: {project_description}

Technology Stack:
- Frontend: {frontend}
- Preferred Language: {language}

Technical Requirements:
1. Create a functional frontend application with all necessary components
2. Implement proper routing between different pages/views
3. Include state management (using appropriate libraries for {frontend})
4. Implement form validation and error handling
5. Create responsive UI that works on mobile and desktop
6. Add mock API integration (with proper loading and error states)
7. Include authentication UI components if applicable
8. Use modern best practices for {frontend}

{coding_guidelines}
{language_specific}

Please provide all necessary files for a working frontend implementation, including:
1. Project structure (show directory hierarchy)
2. Main application file
3. Key components
4. Routing configuration
5. State management setup
6. API integration service
7. Styling (CSS, SCSS, or styled components)
8. Package.json or equivalent with dependencies

The code should be complete enough to run after copying, not just code snippets.
"""
    elif component == "backend":
        prompt = f"""
Generate complete, production-ready backend code for a {architecture} application with the following specification:

Project Description: {project_description}

Technology Stack:
- Backend: {backend}
- Database: {database}
- Preferred Language: {language}

Technical Requirements:
1. Create a complete API server with RESTful endpoints
2. Implement proper middleware (authentication, logging, error handling)
3. Set up database connection and models
4. Implement business logic in service layer
5. Include input validation and error handling
6. Add authentication and authorization if applicable
7. Follow security best practices
8. Structure code for maintainability and scalability

{coding_guidelines}
{language_specific}
{framework_specific}
{database_specific}

Please provide all necessary files for a working backend implementation, including:
1. Project structure (show directory hierarchy)
2. Main application file
3. Route definitions
4. Controller/handler functions
5. Service layer with business logic
6. Data models
7. Middleware implementations
8. Configuration files
9. Package.json/requirements.txt with dependencies

The code should be complete enough to run after copying with minimal setup, not just code snippets.
"""
    elif component == "database":
        prompt = f"""
Generate complete database schema and data access code for a {architecture} application with the following specification:

Project Description: {project_description}

Technology Stack:
- Database: {database}
- Backend: {backend}
- Preferred Language: {language}

Technical Requirements:
1. Design a complete database schema with appropriate tables/collections
2. Include proper relationships, keys, and indexes
3. Implement data access patterns (repository pattern, DAOs, etc.)
4. Include sample CRUD operations
5. Add migration scripts where appropriate
6. Implement proper error handling for database operations
7. Follow performance best practices for {database}
8. Include data validation

{coding_guidelines}
{language_specific}
{database_specific}

Please provide:
1. Complete schema definition (SQL scripts, ORM models, NoSQL schemas)
2. Migration scripts or setup instructions
3. Repository/DAO layer implementation
4. Sample queries for common operations
5. Database configuration
6. Initialization/seeding scripts with sample data
7. Utility functions for common database operations

The code should be complete and functional, not just code snippets.
"""
    else:  # all components
        prompt = f"""
Generate complete, production-ready code for a {architecture} application with the following specification:

Project Description: {project_description}

Technology Stack:
- Frontend: {frontend}
- Backend: {backend}
- Database: {database}
- Preferred Language: {language}

Technical Requirements:
1. Create a fully functional application with frontend and backend
2. Implement proper communication between frontend and backend
3. Set up database integration with models/schema
4. Include authentication and authorization
5. Implement proper error handling throughout
6. Follow security best practices
7. Structure code for maintainability and scalability
8. Include comprehensive README with setup instructions

{coding_guidelines}
{language_specific}
{framework_specific}
{database_specific}

Focus on creating a working application rather than just sample code. Please provide:
1. Project structure for both frontend and backend (show directory hierarchy)
2. Main application files for both parts
3. Key frontend components and pages
4. Backend API routes and handlers
5. Database schema and integration
6. Authentication implementation
7. State management (frontend)
8. Service layer with business logic (backend)
9. All necessary configuration files
10. Package.json/requirements.txt with dependencies

The application should follow a clear architecture pattern appropriate for the selected stack.
"""
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # Lower temperature for more focused code generation
        "max_tokens": 4096
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        code = result["choices"][0]["message"]["content"]
        
        # Post-process the code for better readability
        code = code.replace("```python", "```python\n")
        code = code.replace("```javascript", "```javascript\n")
        code = code.replace("```typescript", "```typescript\n")
        code = code.replace("```java", "```java\n")
        code = code.replace("```go", "```go\n")
        code = code.replace("```sql", "```sql\n")
        
        return code
    except Exception as e:
        return f"Error generating code: {str(e)}\n\nPlease check your API key and try again. If the problem persists, you may need to increase your token limit or try generating smaller components separately."
# Function to create enhanced architecture diagrams
def create_architecture_diagram(architecture_name, frontend, backend, database, project_description=""):
    graph = graphviz.Digraph(format="png")
    graph.attr(fontname="Arial", rankdir="TB", splines="ortho", nodesep="0.5", ranksep="0.75")
    graph.attr("node", fontname="Arial", fontsize="12", shape="box", style="rounded,filled", margin="0.2,0.1")
    graph.attr("edge", fontname="Arial", fontsize="10")
    
    # Define a color palette
    colors = {
        "user": "#E3F2FD",        # Light blue
        "client": "#E8F5E9",      # Light green
        "frontend": "#BBDEFB",    # Medium blue
        "backend": "#C8E6C9",     # Medium green
        "service": "#DCEDC8",     # Light lime
        "database": "#FFF9C4",    # Light yellow
        "messaging": "#FFE0B2",   # Light orange
        "cache": "#F8BBD0",       # Light pink
        "cloud": "#D1C4E9",       # Light purple
        "gateway": "#B3E5FC",     # Sky blue
        "security": "#FFCCBC",    # Light red
        "monitor": "#CFD8DC",     # Light gray
        "api": "#B2DFDB"          # Teal
    }
    
    # Extract project-specific keywords to customize diagram
    keywords = {}
    if project_description:
        doc = nlp(project_description.lower())
        # Check for domain-specific keywords
        domains = {
            "ecommerce": ["shop", "store", "ecommerce", "product", "cart", "checkout", "payment", "order"],
            "social": ["social", "post", "friend", "follow", "profile", "user", "comment", "like", "share"],
            "finance": ["payment", "transaction", "account", "finance", "banking", "invoice", "credit", "debit"],
            "education": ["course", "student", "teacher", "learn", "education", "class", "lesson", "quiz"],
            "healthcare": ["patient", "doctor", "health", "medical", "appointment", "diagnosis", "treatment"],
            "iot": ["device", "sensor", "iot", "monitor", "smart", "connected"],
            "analytics": ["analytics", "dashboard", "report", "metric", "chart", "visualization", "insight"]
        }
        
        for domain, domain_keywords in domains.items():
            for kw in domain_keywords:
                if kw in project_description.lower():
                    keywords[domain] = True
    
    # Customize diagram title based on project description and technologies
    title_prefix = ""
    for domain in keywords:
        title_prefix = domain.capitalize() + " "
    
    if architecture_name == "MVC (Model-View-Controller)":
        # MVC Architecture with specified technologies
        graph.attr(label=f"{title_prefix}MVC Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # User and browser
        graph.node("User", shape="person", style="filled", fillcolor=colors["user"], fontcolor="black")
        graph.node("Browser", shape="component", style="filled", fillcolor=colors["client"], fontcolor="black")
        
        # Frontend (View)
        graph.node(f"View\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        
        # Backend (Controller)
        graph.node(f"Controller\n({backend})", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        
        # Database (Model)
        graph.node(f"Model", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
        graph.node(f"Database\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        
        # Domain-specific customizations
        if "ecommerce" in keywords:
            graph.node("Product Catalog", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.node("Shopping Cart", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.node("Payment Gateway", shape="hexagon", style="filled", fillcolor=colors["gateway"], fontcolor="black")
            
            graph.edge(f"Controller\n({backend})", "Product Catalog")
            graph.edge(f"Controller\n({backend})", "Shopping Cart")
            graph.edge(f"Controller\n({backend})", "Payment Gateway", style="dashed")
            
        elif "social" in keywords:
            graph.node("User Profiles", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.node("Content Feed", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            
            graph.edge(f"Controller\n({backend})", "User Profiles")
            graph.edge(f"Controller\n({backend})", "Content Feed")
            
        elif "analytics" in keywords:
            graph.node("Data Processor", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.node("Visualization Engine", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            
            graph.edge(f"Controller\n({backend})", "Data Processor")
            graph.edge("Data Processor", "Visualization Engine")
            graph.edge("Visualization Engine", f"View\n({frontend})")
        
        # Auth and cache - standard for most MVC apps
        graph.node("Authentication", shape="hexagon", style="filled", fillcolor=colors["security"], fontcolor="black")
        graph.node("Cache", shape="cylinder", style="filled", fillcolor=colors["cache"], fontcolor="black")
        
        # Connections
        graph.edge("User", "Browser", label="interacts")
        graph.edge("Browser", f"View\n({frontend})", label="renders")
        graph.edge(f"View\n({frontend})", f"Controller\n({backend})", label="requests")
        graph.edge(f"Controller\n({backend})", "Authentication", style="dashed", label="verifies")
        graph.edge(f"Controller\n({backend})", f"Model", label="updates")
        graph.edge(f"Model", f"Database\n({database})", label="persists")
        graph.edge(f"Controller\n({backend})", "Cache", style="dashed", label="uses")
        graph.edge(f"Model", f"View\n({frontend})", label="updates")
        
    elif architecture_name == "Microservices Architecture":
        # Microservices with specified technologies
        graph.attr(label=f"{title_prefix}Microservices Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Client and gateway
        graph.node("Client\n(Browser/Mobile)", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        graph.node(f"Frontend\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        graph.node("API Gateway", shape="hexagon", style="filled", fillcolor=colors["gateway"], fontcolor="black")
        
        # Authentication and service discovery
        graph.node("Authentication", shape="hexagon", style="filled", fillcolor=colors["security"], fontcolor="black")
        graph.node("Service Registry", shape="diamond", style="filled", fillcolor=colors["monitor"], fontcolor="black")
        
        # Define microservices based on project type
        services = []
        
        if "ecommerce" in keywords:
            services = [
                {"name": "User Service", "db": "User DB"},
                {"name": "Product Service", "db": "Product DB"},
                {"name": "Order Service", "db": "Order DB"},
                {"name": "Payment Service", "db": "Payment DB"},
                {"name": "Inventory Service", "db": "Inventory DB"}
            ]
        elif "social" in keywords:
            services = [
                {"name": "User Service", "db": "User DB"},
                {"name": "Post Service", "db": "Content DB"},
                {"name": "Notification Service", "db": "Notification DB"},
                {"name": "Messaging Service", "db": "Message DB"},
                {"name": "Analytics Service", "db": "Analytics DB"}
            ]
        elif "finance" in keywords:
            services = [
                {"name": "Account Service", "db": "Account DB"},
                {"name": "Transaction Service", "db": "Transaction DB"},
                {"name": "Authentication Service", "db": "Auth DB"},
                {"name": "Reporting Service", "db": "Report DB"},
                {"name": "Notification Service", "db": "Notification DB"}
            ]
        elif "healthcare" in keywords:
            services = [
                {"name": "Patient Service", "db": "Patient DB"},
                {"name": "Appointment Service", "db": "Appointment DB"},
                {"name": "Medical Record Service", "db": "Records DB"},
                {"name": "Billing Service", "db": "Billing DB"},
                {"name": "Notification Service", "db": "Notification DB"}
            ]
        elif "iot" in keywords:
            services = [
                {"name": "Device Service", "db": "Device DB"},
                {"name": "Data Ingestion", "db": "Telemetry DB"},
                {"name": "Rule Engine", "db": "Rules DB"},
                {"name": "Alert Service", "db": "Alert DB"},
                {"name": "Analytics Service", "db": "Analytics DB"}
            ]
        else:
            # Default services if no specific domain detected
            services = [
                {"name": "User Service", "db": "User DB"},
                {"name": "Content Service", "db": "Content DB"},
                {"name": "Notification Service", "db": "Notification DB"},
                {"name": "API Service", "db": "API DB"}
            ]
        
        # Create services and databases with equal spacing
        with graph.subgraph() as s:
            s.attr(rank="same")
            for i, service in enumerate(services):
                service_name = f"{service['name']}\n({backend})"
                graph.node(service_name, shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
                db_name = f"{service['db']}\n({database})"
                graph.node(db_name, shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
                graph.edge(service_name, db_name)
        
        # Message broker and monitoring
        graph.node("Message Broker", shape="hexagon", style="filled", fillcolor=colors["messaging"], fontcolor="black")
        graph.node("Monitoring & Logging", shape="box", style="filled", fillcolor=colors["monitor"], fontcolor="black")
        
        # Connect components
        graph.edge("Client\n(Browser/Mobile)", f"Frontend\n({frontend})")
        graph.edge(f"Frontend\n({frontend})", "API Gateway")
        graph.edge("API Gateway", "Authentication", style="dashed")
        
        for service in services:
            service_name = f"{service['name']}\n({backend})"
            graph.edge("API Gateway", service_name)
            graph.edge(service_name, "Service Registry", dir="both", style="dashed")
            graph.edge(service_name, "Message Broker", dir="both", style="dashed")
            graph.edge(service_name, "Monitoring & Logging", style="dashed", dir="both")
            
        graph.edge("Message Broker", "Monitoring & Logging", style="dashed")
        
    elif architecture_name == "Event-Driven Architecture":
        # Event-driven architecture
        graph.attr(label=f"{title_prefix}Event-Driven Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Client and frontend
        graph.node("Client\n(Browser/Mobile)", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        graph.node(f"Frontend\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        
        # Event bus and message broker
        graph.node("Event Bus", shape="hexagon", style="filled", fillcolor=colors["messaging"], fontcolor="black")
        
        # Define producers and consumers based on project type
        producers = []
        consumers = []
        
        if "ecommerce" in keywords:
            producers = ["Order Events", "Inventory Events", "Payment Events", "User Events"]
            consumers = ["Order Processor", "Inventory Manager", "Payment Gateway", "Notification Service", "Analytics Service"]
        elif "social" in keywords:
            producers = ["User Events", "Content Events", "Interaction Events", "Notification Triggers"]
            consumers = ["Content Feed", "Notification Service", "Analytics Service", "Recommendation Engine"]
        elif "finance" in keywords:
            producers = ["Transaction Events", "Account Events", "Market Data Events", "User Events"]
            consumers = ["Transaction Processor", "Fraud Detection", "Reporting Service", "Notification Service"]
        elif "iot" in keywords:
            producers = ["Device Events", "Sensor Data", "Command Events", "System Events"]
            consumers = ["Device Manager", "Data Processor", "Alert Manager", "Analytics Engine", "Visualization Service"]
        else:
            # Default producers/consumers if no specific domain detected
            producers = ["User Events", "System Events", "External Events"]
            consumers = ["Notification Service", "Analytics Service", "Data Service", "Logging Service"]
        
        # Create producers
        for producer in producers:
            producer_name = f"{producer}\n({backend})"
            graph.node(producer_name, shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
            graph.edge(producer_name, "Event Bus", label="publishes")
        
        # Create consumers
        for consumer in consumers:
            consumer_name = f"{consumer}\n({backend})"
            graph.node(consumer_name, shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.edge("Event Bus", consumer_name, label="subscribes")
        
        # Database and cache
        graph.node(f"Database\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        graph.node("Cache", shape="cylinder", style="filled", fillcolor=colors["cache"], fontcolor="black")
        
        # Connect components
        graph.edge("Client\n(Browser/Mobile)", f"Frontend\n({frontend})")
        graph.edge(f"Frontend\n({frontend})", "Event Bus", dir="both")
        
        # Connect services to database - customize based on domain
        if "ecommerce" in keywords:
            graph.edge("Order Processor\n(backend)", f"Database\n({database})")
            graph.edge("Inventory Manager\n(backend)", f"Database\n({database})")
            graph.edge("Notification Service\n(backend)", "Cache")
        elif "social" in keywords:
            graph.edge("Content Feed\n(backend)", f"Database\n({database})")
            graph.edge("Recommendation Engine\n(backend)", f"Database\n({database})")
            graph.edge("Notification Service\n(backend)", "Cache")
        else:
            # Default connections
            graph.edge(f"{consumers[0]}\n({backend})", f"Database\n({database})")
            if len(consumers) > 1:
                graph.edge(f"{consumers[1]}\n({backend})", f"Database\n({database})", style="dashed")
            if len(consumers) > 2:
                graph.edge(f"{consumers[2]}\n({backend})", "Cache", style="dashed")
        
    elif architecture_name == "Serverless Architecture":
        # Serverless architecture
        graph.attr(label=f"{title_prefix}Serverless Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Client and frontend
        graph.node("Client\n(Browser/Mobile)", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        graph.node(f"Frontend\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        
        # API Gateway
        graph.node("API Gateway", shape="hexagon", style="filled", fillcolor=colors["gateway"], fontcolor="black")
        
        # Cloud provider
        graph.node("Cloud Provider", shape="cloud", style="filled", fillcolor=colors["cloud"], fontcolor="black")
        
        # Define functions based on project type
        functions = []
        
        if "ecommerce" in keywords:
            functions = ["Auth Function", "Product Function", "Cart Function", "Order Function", "Payment Function", "Inventory Function"]
        elif "social" in keywords:
            functions = ["Auth Function", "User Profile Function", "Content Function", "Feed Function", "Notification Function"]
        elif "finance" in keywords:
            functions = ["Auth Function", "Account Function", "Transaction Function", "Reporting Function", "Notification Function"]
        elif "analytics" in keywords:
            functions = ["Data Ingestion", "Data Processing", "Analysis Function", "Visualization Function", "Export Function"]
        elif "iot" in keywords:
            functions = ["Device Registration", "Data Ingestion", "Data Processing", "Alert Function", "Visualization Function"]
        else:
            # Default functions if no specific domain detected
            functions = ["Auth Function", "User Function", "Content Function", "API Function", "Notification Function"]
        
        # Create cloud services for the architecture
        cloud_services = []
        if "ecommerce" in keywords:
            cloud_services = ["Payment Service", "Storage Service", "CDN"]
        elif "social" in keywords:
            cloud_services = ["Storage Service", "CDN", "Push Notification Service"]
        elif "analytics" in keywords:
            cloud_services = ["Data Lake", "Analytics Service", "ML Service"]
        elif "iot" in keywords:
            cloud_services = ["IoT Hub", "Stream Analytics", "Time Series DB"]
        else:
            cloud_services = ["Storage Service", "Notification Service"]
            
        # Create services
        for service in cloud_services:
            service_name = f"{service}"
            graph.node(service_name, shape="hexagon", style="filled", fillcolor=colors["cloud"], fontcolor="black")
            graph.edge("Cloud Provider", service_name, style="dashed")
        
        # Create functions
        for function in functions:
            function_name = f"{function}\n({backend})"
            graph.node(function_name, shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
            graph.edge("API Gateway", function_name)
            graph.edge("Cloud Provider", function_name, style="dashed")
            
            # Connect functions to cloud services
            if "Payment" in function and "Payment Service" in cloud_services:
                graph.edge(function_name, "Payment Service", style="dashed")
            if ("Product" in function or "Content" in function) and "Storage Service" in cloud_services:
                graph.edge(function_name, "Storage Service", style="dashed")
            if "Notification" in function and "Push Notification Service" in cloud_services:
                graph.edge(function_name, "Push Notification Service", style="dashed")
        
        # Database and storage services
        graph.node(f"Database\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        graph.node("Blob Storage", shape="cylinder", style="filled", fillcolor=colors["cache"], fontcolor="black")
        graph.node("Queue Service", shape="hexagon", style="filled", fillcolor=colors["messaging"], fontcolor="black")
        
        # Connect components
        graph.edge("Client\n(Browser/Mobile)", f"Frontend\n({frontend})")
        graph.edge(f"Frontend\n({frontend})", "API Gateway")
        
        # Connect functions to services based on domain
        if "ecommerce" in keywords:
            graph.edge(f"User Function\n({backend})", f"Database\n({database})")
            graph.edge(f"Product Function\n({backend})", f"Database\n({database})")
            graph.edge(f"Order Function\n({backend})", f"Database\n({database})")
            graph.edge(f"Payment Function\n({backend})", "Queue Service")
            graph.edge(f"Product Function\n({backend})", "Blob Storage", style="dashed")
            
            if "CDN" in cloud_services:
                graph.edge("Product Function\n(backend)", "CDN", style="dashed")
        else:
            # Default connections
            for i, function in enumerate(functions):
                if i < 3:  # Connect first 3 functions to database
                    graph.edge(f"{function}\n({backend})", f"Database\n({database})")
                if i == 3:  # Connect 4th function to queue
                    graph.edge(f"{function}\n({backend})", "Queue Service")
                if i == 4:  # Connect 5th function to blob storage
                    graph.edge(f"{function}\n({backend})", "Blob Storage", style="dashed")
        
    elif architecture_name == "Batch Processing Architecture":
        # Batch processing architecture
        graph.attr(label=f"{title_prefix}Batch Processing Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Data sources
        data_sources = ["Source DB", "File Storage", "External APIs"]
        for i, source in enumerate(data_sources):
            graph.node(source, shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        
        # ETL pipeline
        graph.node("Data Extractor", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        graph.node("Data Transformer", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        graph.node("Data Loader", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        graph.node("Scheduler", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
        
        # Data warehouse
        graph.node(f"Data Warehouse\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        
        # Analytics and reporting
        graph.node("Analytics Engine", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
        graph.node("Report Generator", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
        graph.node(f"Dashboard\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        
        # Connections
        for source in data_sources:
            graph.edge(source, "Data Extractor", label="extract")
        
        graph.edge("Scheduler", "Data Extractor", label="triggers")
        graph.edge("Data Extractor", "Data Transformer", label="raw data")
        graph.edge("Data Transformer", "Data Loader", label="transformed")
        graph.edge("Data Loader", f"Data Warehouse\n({database})", label="load")
        graph.edge(f"Data Warehouse\n({database})", "Analytics Engine", label="query")
        graph.edge("Analytics Engine", "Report Generator", label="metrics")
        graph.edge("Report Generator", f"Dashboard\n({frontend})", label="visualize")
        
        # Add some domain-specific nodes if applicable
        if "ecommerce" in keywords:
            graph.node("Sales Reports", shape="note", style="filled", fillcolor=colors["frontend"], fontcolor="black")
            graph.node("Inventory Reports", shape="note", style="filled", fillcolor=colors["frontend"], fontcolor="black")
            graph.edge("Report Generator", "Sales Reports")
            graph.edge("Report Generator", "Inventory Reports")
        elif "finance" in keywords:
            graph.node("Financial Reports", shape="note", style="filled", fillcolor=colors["frontend"], fontcolor="black")
            graph.node("Compliance Reports", shape="note", style="filled", fillcolor=colors["frontend"], fontcolor="black")
            graph.edge("Report Generator", "Financial Reports")
            graph.edge("Report Generator", "Compliance Reports")
    
    elif architecture_name == "Client-Server Architecture":
        # Client-Server architecture for mobile apps
        graph.attr(label=f"{title_prefix}Client-Server Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Client devices
        graph.node("Mobile App", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        graph.node("Web Client", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        
        # Server components
        graph.node(f"API Server\n({backend})", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        graph.node("Authentication", shape="hexagon", style="filled", fillcolor=colors["security"], fontcolor="black")
        graph.node(f"Database\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        graph.node("Cache", shape="cylinder", style="filled", fillcolor=colors["cache"], fontcolor="black")
        
        # Connect components
        graph.edge("Mobile App", f"API Server\n({backend})", label="HTTP/REST")
        graph.edge("Web Client", f"API Server\n({backend})", label="HTTP/REST")
        graph.edge(f"API Server\n({backend})", "Authentication", style="dashed", label="verifies")
        graph.edge(f"API Server\n({backend})", f"Database\n({database})", label="CRUD")
        graph.edge(f"API Server\n({backend})", "Cache", style="dashed", label="uses")
        
        # Add domain-specific components
        if "ecommerce" in keywords:
            graph.node("Payment Gateway", shape="hexagon", style="filled", fillcolor=colors["gateway"], fontcolor="black")
            graph.node("Push Notifications", shape="component", style="filled", fillcolor=colors["service"], fontcolor="black")
            graph.edge(f"API Server\n({backend})", "Payment Gateway", style="dashed")
            graph.edge(f"API Server\n({backend})", "Push Notifications", style="dashed")
            graph.edge("Push Notifications", "Mobile App", style="dashed", dir="back")
        elif "social" in keywords:
            graph.node("WebSockets", shape="hexagon", style="filled", fillcolor=colors["messaging"], fontcolor="black")
            graph.node("CDN", shape="hexagon", style="filled", fillcolor=colors["cloud"], fontcolor="black")
            graph.edge(f"API Server\n({backend})", "WebSockets", style="dashed")
            graph.edge("WebSockets", "Mobile App", style="dashed", dir="both", label="real-time")
            graph.edge("WebSockets", "Web Client", style="dashed", dir="both", label="real-time")
            graph.edge("CDN", "Mobile App", label="media")
            graph.edge("CDN", "Web Client", label="media")
    
    elif architecture_name == "Peer-to-Peer Architecture":
        # P2P architecture
        graph.attr(label=f"{title_prefix}Peer-to-Peer Architecture with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Peer nodes
        for i in range(1, 6):
            graph.node(f"Peer {i}", shape="circle", style="filled", fillcolor=colors["client"], fontcolor="black")
        
        # Central components (if any)
        graph.node("Discovery Server", shape="hexagon", style="filled", fillcolor=colors["gateway"], fontcolor="black")
        graph.node("Bootstrap Node", shape="hexagon", style="filled", fillcolor=colors["cloud"], fontcolor="black")
        
        # Connect peers in a mesh
        graph.edge("Peer 1", "Peer 2", dir="both")
        graph.edge("Peer 1", "Peer 3", dir="both")
        graph.edge("Peer 2", "Peer 4", dir="both")
        graph.edge("Peer 3", "Peer 5", dir="both")
        graph.edge("Peer 4", "Peer 5", dir="both")
        graph.edge("Peer 2", "Peer 3", dir="both", style="dashed")
        graph.edge("Peer 4", "Peer 1", dir="both", style="dashed")
        
        # Connect to central components
        for i in range(1, 6):
            graph.edge(f"Peer {i}", "Discovery Server", style="dashed", dir="both")
            
        graph.edge("Bootstrap Node", "Discovery Server")
        graph.edge("Bootstrap Node", "Peer 1", style="dashed")
        
        # Add domain-specific elements
        if "blockchain" in keywords:
            graph.node("Blockchain", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
            for i in range(1, 6):
                graph.edge(f"Peer {i}", "Blockchain", dir="both", style="dashed")
    
    else:
        # Generic architecture diagram
        graph.attr(label=f"{title_prefix}{architecture_name} with {frontend}, {backend}, and {database}", labelloc="t", fontsize="16")
        
        # Client and frontend
        graph.node("Client", shape="box", style="filled", fillcolor=colors["client"], fontcolor="black")
        graph.node(f"Frontend\n({frontend})", shape="component", style="filled", fillcolor=colors["frontend"], fontcolor="black")
        
        # Backend and database
        graph.node(f"Backend\n({backend})", shape="component", style="filled", fillcolor=colors["backend"], fontcolor="black")
        graph.node(f"Database\n({database})", shape="cylinder", style="filled", fillcolor=colors["database"], fontcolor="black")
        
        # Connect basic components
        graph.edge("Client", f"Frontend\n({frontend})")
        graph.edge(f"Frontend\n({frontend})", f"Backend\n({backend})")
        graph.edge(f"Backend\n({backend})", f"Database\n({database})")
        
        # Add some common elements to make it more descriptive
        graph.node("API Layer", shape="component", style="filled", fillcolor=colors["api"], fontcolor="black")
        graph.node("Authentication", shape="hexagon", style="filled", fillcolor=colors["security"], fontcolor="black")
        
        graph.edge(f"Frontend\n({frontend})", "API Layer")
        graph.edge("API Layer", f"Backend\n({backend})")
        graph.edge("API Layer", "Authentication", style="dashed")
    
    return graph
def show_architecture_visualizer_sidebar():
    st.sidebar.title("Code Architecture Visualizer")
    st.sidebar.write("Analyze and visualize your code architecture")
    
    with st.sidebar.expander("Visualize Code Architecture", expanded=False):
        code = st.text_area("Code Input", height=250, placeholder="Paste your code here to analyze its architecture...")
        
        col1, col2 = st.columns(2)
        with col1:
            parse_method = st.radio(
                "Parsing Method",
                ["Basic Parser", "LLM Analysis (Groq)"],
                help="Choose how to analyze the code relationships"
            )
        with col2:
            diagram_type = st.radio(
                "Diagram Type",
                ["NetworkX", "Mermaid"],
                help="Choose the diagram visualization type"
            )
        if st.button("Generate Diagram"):
            if not code:
                st.error("Please paste some code to analyze.")
            else:
                with st.spinner("Analyzing code and generating diagram..."):
                    try:
                        # Extract basic code components
                        components = extract_python_components(code)
                        
                        # Get additional analysis if LLM option selected
                        if parse_method == "LLM Analysis (Groq)":
                            llm_analysis = analyze_code_with_llm(code)
                            # Merge basic components with LLM analysis
                            analysis = {**components, **llm_analysis}
                        else:
                            # Create relationships based on basic parsing
                            analysis = components
                            analysis['relationships'] = []
                        
                        # Generate the selected diagram type
                        if diagram_type == "Mermaid":
                            mermaid_code = generate_mermaid_diagram(analysis)
                            st.subheader("Mermaid Diagram")
                            st.code(mermaid_code, language="mermaid")
                            
                        else:  # NetworkX
                            fig = generate_networkx_diagram(analysis)
                            st.subheader("NetworkX Diagram")
                            st.pyplot(fig)
                        
                        # Display code components
                        with st.expander("Code Components"):
                            st.subheader("Classes")
                            if components['classes']:
                                for cls in components['classes']:
                                    st.write(f"**{cls['name']}**")
                                    if cls['parents']:
                                        st.write(f"Inherits from: {', '.join(cls['parents'])}")
                                    if cls['methods']:
                                        st.write(f"Methods: {', '.join(cls['methods'])}")
                                    if cls['attributes']:
                                        st.write(f"Attributes: {', '.join(cls['attributes'])}")
                                    st.write("---")
                            else:
                                st.write("No classes found.")
                            
                            st.subheader("Functions")
                            st.write(", ".join(components['functions']) if components['functions'] else "None found")
                            
                            st.subheader("Imports")
                            st.write(", ".join(components['imports']) if components['imports'] else "None found")
                        
                        # Display raw LLM analysis if used
                        if parse_method == "LLM Analysis (Groq)" and "raw_analysis" in analysis:
                            with st.expander("Raw LLM Analysis"):
                                st.write(analysis["raw_analysis"])
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please check your code and try again.")

# Function to get tech options for selection based on chosen language and architecture
def get_tech_options(architecture_name, language="Python"):
    """
    Returns appropriate technology options for frontend, backend, and database
    based on the selected architecture and programming language.
    
    Args:
        architecture_name (str): The name of the selected architecture
        language (str): The selected programming language
    
    Returns:
        tuple: Lists of (frontend_options, backend_options, database_options)
    """
    # First check if the language-specific options exist for this architecture
    if language in tech_stacks_by_language and architecture_name in tech_stacks_by_language[language]:
        tech_options = tech_stacks_by_language[language][architecture_name]
        
        # Get frontend options
        frontend_options = tech_options.get("frontend", [])
        if not frontend_options and "client" in tech_options:
            frontend_options = tech_options.get("client", [])
            
        # Get backend options
        backend_options = tech_options.get("backend", [])
        if not backend_options and "server" in tech_options:
            backend_options = tech_options.get("server", [])
            
        # Get database options
        database_options = tech_options.get("database", [])
        
        # If we have valid options, return them
        if frontend_options and backend_options and database_options:
            return frontend_options, backend_options, database_options
    
    # If we don't have language-specific options, check general tech stacks
    if architecture_name in tech_stacks:
        tech_options = tech_stacks[architecture_name]
        
        # Get frontend options
        frontend_options = tech_options.get("frontend", [])
        if not frontend_options and "client" in tech_options:
            frontend_options = tech_options.get("client", [])
            
        # Get backend options
        backend_options = tech_options.get("backend", [])
        if not backend_options and "server" in tech_options:
            backend_options = tech_options.get("server", [])
            
        # Get database options
        database_options = tech_options.get("database", [])
        
        # If we have valid options, return them
        if frontend_options and backend_options and database_options:
            return frontend_options, backend_options, database_options
    
    # If we still don't have options, provide defaults based on selected language
    default_options = {
        "Python": {
            "frontend": ["React", "Vue.js", "Angular", "Next.js"],
            "backend": ["Django", "Flask", "FastAPI", "Pyramid"],
            "database": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis"]
        },
        "JavaScript": {
            "frontend": ["React", "Vue.js", "Angular", "Svelte", "Next.js"],
            "backend": ["Express.js", "NestJS", "Koa", "Fastify", "Hapi"],
            "database": ["MongoDB", "PostgreSQL", "MySQL", "Redis", "Firebase"]
        },
        "TypeScript": {
            "frontend": ["React with TypeScript", "Angular", "Vue with TypeScript", "Next.js"],
            "backend": ["NestJS", "Express with TypeScript", "Fastify with TypeScript"],
            "database": ["PostgreSQL", "MongoDB", "MySQL", "Redis"]
        },
        "Java": {
            "frontend": ["React", "Angular", "Vue.js"],
            "backend": ["Spring Boot", "Spring MVC", "Quarkus", "Micronaut", "Jakarta EE"],
            "database": ["PostgreSQL", "MySQL", "Oracle", "MongoDB", "Cassandra"]
        },
        "Go": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Gin", "Echo", "Fiber", "Revel", "Buffalo"],
            "database": ["PostgreSQL", "MySQL", "MongoDB", "Redis"]
        }
    }
    
    # Get default options for the language if available, or provide universal defaults
    if language in default_options:
        return (
            default_options[language]["frontend"], 
            default_options[language]["backend"], 
            default_options[language]["database"]
        )
    else:
        # Universal defaults as a last resort
        return (
            ["React", "Vue.js", "Angular"], 
            ["Express.js", "Django", "Spring Boot", "Flask"], 
            ["PostgreSQL", "MySQL", "MongoDB", "SQLite"]
        )
    
    
# Main Streamlit app
# Main Streamlit app
def main():
    st.title("🏗️ Architron - Software Architecture Assistant")
    st.markdown("""
    This tool helps you design and plan software architecture based on your requirements. 
    Explore different architecture patterns, technology stacks, and deployment options.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Tool", [
        "Architecture Planner", 
        "Code Architecture Visualizer"
    ])
    
    # Determine which page to show
    if page == "Architecture Planner":
        show_architecture_planner()
    elif page == "Code Architecture Visualizer":
        show_architecture_visualizer()

def show_architecture_planner():
    # Sidebar for project settings
    st.sidebar.title("Project Settings")
    
    # Use templates or custom description
    use_template = st.sidebar.checkbox("Use project template", value=True)
    
    if use_template:
        template_name = st.sidebar.selectbox("Select project type", list(project_templates.keys()))
        project_description = project_templates[template_name]
        st.sidebar.text_area("Project description", project_description, height=150, disabled=True)
    else:
        project_description = st.sidebar.text_area("Describe your project requirements", 
            "Build a web application for managing inventory, orders, and customer data for a small business.", height=150)
    
    # Extract project type from description
    doc = nlp(project_description.lower())
    
    # Detect keywords in description to suggest architecture
    architecture_keywords = {
        "web application": ["web", "website", "webapp", "site", "browser"],
        "real-time system": ["real-time", "realtime", "live", "streaming", "instant", "notifications", "chat"],
        "scalable system": ["scale", "scalable", "microservice", "distributed", "high-volume", "traffic"],
        "data processing": ["data", "analytics", "reports", "processing", "etl", "transform", "pipeline"],
        "ai model": ["ai", "ml", "model", "prediction", "machine learning", "neural", "train"],
        "mobile": ["mobile", "app", "android", "ios", "smartphone"],
        "distributed": ["distributed", "blockchain", "p2p", "peer", "decentralized"]
    }
    
    # Count occurrences of each architecture's keywords
    architecture_scores = defaultdict(int)
    for arch, keywords in architecture_keywords.items():
        for token in doc:
            if token.text in keywords or any(kw in token.text for kw in keywords):
                architecture_scores[arch] += 1
        
        # Check for phrases
        for kw in keywords:
            if kw in project_description.lower():
                architecture_scores[arch] += 1
    
    # Get the highest scoring architecture types (might be multiple)
    max_score = max(architecture_scores.values()) if architecture_scores else 0
    suggested_archs = [arch for arch, score in architecture_scores.items() if score == max_score]
    suggested_arch = suggested_archs[0] if suggested_archs else "web application"  # Default to web app
    
    # Architecture selection
    st.sidebar.subheader("Architecture Selection")
    selected_arch_type = st.sidebar.selectbox(
        "Select architecture type", 
        list(architecture_types.keys()),
        index=list(architecture_types.keys()).index(suggested_arch) if suggested_arch in architecture_types else 0
    )
    
    # Architecture details
    arch_details = architecture_types[selected_arch_type]
    selected_arch_name = arch_details["name"]
    
    # Programming language selection
    available_languages = list(tech_stacks_by_language.keys())
    selected_language = st.sidebar.selectbox("Select programming language", available_languages)
    
    # Get technology options based on selected architecture and language
    frontend_options, backend_options, database_options = get_tech_options(selected_arch_name, selected_language)
    
    # Technology stack selection
    st.sidebar.subheader("Technology Stack")
    
    # If no options available for the selected language and architecture, show message
    if not frontend_options or not backend_options or not database_options:
        st.sidebar.warning(f"Limited technology options available for {selected_arch_name} with {selected_language}. Showing generic options.")
        # Fall back to default options
        frontend_options = ["React", "Vue.js", "Angular"] if not frontend_options else frontend_options
        backend_options = [f"{selected_language} Backend", "Express", "Django"] if not backend_options else backend_options
        database_options = ["PostgreSQL", "MongoDB", "MySQL"] if not database_options else database_options
    
    selected_frontend = st.sidebar.selectbox("Frontend", frontend_options)
    selected_backend = st.sidebar.selectbox("Backend", backend_options)
    selected_database = st.sidebar.selectbox("Database", database_options)
    
    # Deployment scale selection
    st.sidebar.subheader("Deployment Settings")
    deployment_scale = st.sidebar.radio("Deployment Scale", ["small", "medium", "large"])
    estimated_cost = estimate_deployment_cost(selected_arch_name, deployment_scale)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Architecture Overview", "Technology Stack", "Deployment", "Project Roadmap", "          "])
    
    with tab1:
        st.header("Architecture Overview")
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader(f"{selected_arch_name}")
            st.markdown(f"**Description:** {arch_details['description']}")
            st.markdown(f"**Best for:** {arch_details['best_for']}")
            
            st.subheader("Key Characteristics")
            if selected_arch_name == "MVC (Model-View-Controller)":
                st.markdown("""
                - **Separation of Concerns**: Divides application into Model, View, and Controller.
                - **Data Management**: Model handles data logic and business rules.
                - **User Interface**: View manages the visual presentation.
                - **Business Logic**: Controller processes user input and updates the model.
                - **Scalability**: Moderate, typically monolithic but can be scaled with load balancers.
                - **Complexity**: Low to moderate, good for small to medium applications.
                """)
            elif selected_arch_name == "Microservices Architecture":
                st.markdown("""
                - **Decomposition**: Application divided into small, independent services.
                - **Independence**: Each service can be developed, deployed, and scaled separately.
                - **API Communication**: Services communicate via well-defined APIs.
                - **Database Per Service**: Each service typically has its own database.
                - **Resilience**: Failure in one service doesn't affect the entire system.
                - **Complexity**: High, requires careful service orchestration and monitoring.
                """)
            elif selected_arch_name == "Event-Driven Architecture":
                st.markdown("""
                - **Event-Centric**: System components communicate through events.
                - **Loose Coupling**: Event producers and consumers are decoupled.
                - **Asynchronous**: Operations occur independently without blocking.
                - **Scalability**: Highly scalable, especially for real-time applications.
                - **Resilience**: Events can be replayed if processing fails.
                - **Complexity**: Moderate to high, requires careful event design.
                """)
            elif selected_arch_name == "Serverless Architecture":
                st.markdown("""
                - **Function as a Service (FaaS)**: Code executed in stateless containers.
                - **Event-Triggered**: Functions execute in response to events.
                - **Auto-Scaling**: Automatic scaling based on demand.
                - **Pay-Per-Use**: Only pay for the compute time used.
                - **Managed Infrastructure**: No server management required.
                - **Statelessness**: Functions should be stateless for best results.
                """)
            elif selected_arch_name == "Batch Processing Architecture":
                st.markdown("""
                - **Non-Real-Time**: Processes data in batches at scheduled intervals.
                - **High Volume**: Optimized for processing large amounts of data.
                - **ETL Workflows**: Extract, Transform, Load operations.
                - **Resource Efficiency**: Can utilize resources when demand is low.
                - **Predictable Costs**: Processing happens on a defined schedule.
                - **Complexity**: Moderate, focuses on throughput rather than latency.
                """)
            elif selected_arch_name == "Client-Server Architecture":
                st.markdown("""
               - **Client-Side Focus**: Application logic primarily runs on mobile devices.
               - **Responsive Design**: Adapts to different screen sizes and orientations.
               - **Offline Capability**: Functions with limited or no network connectivity.
               - **Resource Constraints**: Optimized for battery life and limited memory.
               - **Native vs. Cross-Platform**: Choice between platform-specific or cross-platform development.
               - **Security**: Emphasis on local data protection and secure API communication.
               """)
            elif selected_arch_name == "Peer-to-Peer Architecture":
                 st.markdown("""
               - **Decentralization**: Processing and data storage distributed across multiple nodes.
               - **Horizontal Scaling**: Add more machines rather than upgrading existing ones.
               - **Fault Tolerance**: System continues functioning despite partial failures.
               - **Latency Management**: Strategies to handle communication delays between nodes.
               - **Data Consistency**: Mechanisms to maintain data integrity across distributed components.
               - **Complexity**: High, requires sophisticated coordination and synchronization protocols.
               """)
            
        with col2:
            st.subheader("Architecture Diagram")
            try:
                graph = create_architecture_diagram(selected_arch_name, selected_frontend, selected_backend, selected_database)
                st.graphviz_chart(graph)
            except Exception as e:
                st.error(f"Error generating diagram: {str(e)}")
            
            st.info("This diagram shows the key components and relationships in the architecture.")
    
    with tab2:
        st.header("Technology Stack Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Selected Technology Stack")
            st.markdown(f"**Frontend:** {selected_frontend}")
            st.markdown(f"**Backend:** {selected_backend}")
            st.markdown(f"**Database:** {selected_database}")
            st.markdown(f"**Programming Language:** {selected_language}")
            
            st.subheader("Technology Stack Compatibility")
            compatibility_score = random.randint(85, 98)  # In a real app, this would be calculated
            st.progress(compatibility_score/100)
            st.markdown(f"**Compatibility Score:** {compatibility_score}%")
            
            if compatibility_score >= 90:
                st.success("These technologies work extremely well together!")
            elif compatibility_score >= 80:
                st.success("Good technology choices that work well together.")
            else:
                st.warning("These technologies may have some integration challenges.")
        
        with col2:
            st.subheader("Pros and Cons")
            
            # Get pros and cons from the language-specific tech stacks if available
            pros = []
            cons = []
            
            if (selected_language in tech_stacks_by_language and 
                selected_arch_name in tech_stacks_by_language[selected_language]):
                pros = tech_stacks_by_language[selected_language][selected_arch_name].get("pros", [])
                cons = tech_stacks_by_language[selected_language][selected_arch_name].get("cons", [])
            
            # If no pros/cons found, provide some generic ones
            if not pros:
                if selected_arch_name == "MVC (Model-View-Controller)":
                    pros = ["Clear separation of concerns", "Well-established pattern", "Good for web applications"]
                elif selected_arch_name == "Microservices Architecture":
                    pros = ["Independent scaling", "Technology diversity", "Fault isolation"]
                elif selected_arch_name == "Event-Driven Architecture":
                    pros = ["Loose coupling", "Real-time processing", "Good scalability"]
                elif selected_arch_name == "Serverless Architecture":
                    pros = ["No server management", "Auto-scaling", "Pay-per-use pricing"]
            
            if not cons:
                if selected_arch_name == "MVC (Model-View-Controller)":
                    cons = ["Can become monolithic", "Scaling challenges", "Tight coupling between components"]
                elif selected_arch_name == "Microservices Architecture":
                    cons = ["Distributed system complexity", "Service coordination challenges", "Potential network issues"]
                elif selected_arch_name == "Event-Driven Architecture":
                    cons = ["Debugging complexity", "Event versioning challenges", "Eventual consistency issues"]
                elif selected_arch_name == "Serverless Architecture":
                    cons = ["Cold start latency", "Vendor lock-in", "Limited execution duration"]
            
            col_pros, col_cons = st.columns(2)
            
            with col_pros:
                st.markdown("**Pros:**")
                for pro in pros:
                    st.markdown(f"✅ {pro}")
            
            with col_cons:
                st.markdown("**Cons:**")
                for con in cons:
                    st.markdown(f"⚠️ {con}")
            
            st.subheader("Alternative Technologies")
            
            alternative_frontend = [opt for opt in frontend_options if opt != selected_frontend][:2]
            alternative_backend = [opt for opt in backend_options if opt != selected_backend][:2]
            alternative_database = [opt for opt in database_options if opt != selected_database][:2]
            
            st.markdown("You might also consider these alternatives:")
            st.markdown(f"**Frontend:** {', '.join(alternative_frontend)}")
            st.markdown(f"**Backend:** {', '.join(alternative_backend)}")
            st.markdown(f"**Database:** {', '.join(alternative_database)}")
    
    with tab3:
        st.header("Deployment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Deployment Strategy")
            
            if selected_arch_name == "MVC (Model-View-Controller)":
                st.markdown("""
                **Recommended Deployment:**
                - Single server or small cluster deployment
                - Application server for backend
                - Database server
                - Optional load balancer for scaling
                
                **CI/CD Pipeline:**
                - Source control (Git)
                - Build and test automation
                - Deployment to staging and production environments
                - Health checks and rollback procedures
                """)
            elif selected_arch_name == "Microservices Architecture":
                st.markdown("""
                **Recommended Deployment:**
                - Container orchestration with Kubernetes
                - Service mesh for communication
                - API gateway for external access
                - Container registry
                - Distributed database deployment
                
                **CI/CD Pipeline:**
                - Independent pipelines for each service
                - Automated testing and deployment
                - Canary releases
                - Feature flags
                - Service monitoring and alerting
                """)
            elif selected_arch_name == "Event-Driven Architecture":
                st.markdown("""
                **Recommended Deployment:**
                - Message brokers (Kafka, RabbitMQ)
                - Event producers and consumers as services
                - Monitoring and event tracking system
                - Event store for persistence
                
                **CI/CD Pipeline:**
                - Component-based deployment
                - Event schema validation
                - Consumer-driven contract testing
                - Event replay capability
                """)
            elif selected_arch_name == "Serverless Architecture":
                st.markdown("""
                **Recommended Deployment:**
                - Cloud provider's serverless platform
                - Infrastructure as Code (IaC)
                - API Gateway configuration
                - Managed database services
                
                **CI/CD Pipeline:**
                - Function packaging and deployment
                - Automated testing
                - Deployment stages (dev, staging, production)
                - Monitoring and logging setup
                """)
        
        with col2:
            st.subheader("Cost and Resource Estimation")
            
            st.markdown(f"**Estimated Monthly Cost:** {estimated_cost}")
            
            st.markdown("**Resources Required:**")
            if deployment_scale == "small":
                st.markdown("""
                - 1-2 application servers/containers
                - 1 database instance
                - Basic monitoring
                - Suitable for: Development, MVP, small businesses
                """)
            elif deployment_scale == "medium":
                st.markdown("""
                - 3-5 application servers/containers
                - Multiple database instances (primary + replicas)
                - Load balancer
                - Comprehensive monitoring and alerting
                - Suitable for: Small to medium businesses, moderate traffic
                """)
            else:  # large
                st.markdown("""
                - 10+ application servers/containers
                - High-availability database cluster
                - Global CDN
                - Advanced monitoring, alerting, and autoscaling
                - Disaster recovery system
                - Suitable for: Enterprise applications, high traffic
                """)
            
            st.subheader("Cloud Provider Options")
            st.markdown("""
            **AWS:**
            - Strong ecosystem and mature services
            - Global presence
            - Comprehensive feature set
            
            **Azure:**
            - Strong integration with Microsoft products
            - Enterprise-friendly
            - Growing AI and ML capabilities
            
            **Google Cloud:**
            - Strong in data analytics and ML
            - Global network performance
            - Kubernetes-native solutions
            """)
    
    with tab4:
        st.header("Project Roadmap")
        
        roadmap = generate_roadmap(selected_arch_name)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Project Timeline")
            
            # Calculate cumulative time estimates
            cumulative_weeks = 0
            for phase in roadmap:
                duration = phase["duration"]
                if "weeks" in duration:
                    # Extract the range if it exists
                    try:
                        if "-" in duration:
                            min_weeks, max_weeks = map(int, duration.split(" ")[0].split("-"))
                            weeks = (min_weeks + max_weeks) / 2  # Use average for display
                        else:
                            weeks = int(duration.split(" ")[0])
                        
                        starting_week = cumulative_weeks
                        cumulative_weeks += weeks
                        
                        st.markdown(f"**{phase['name']}:** Weeks {starting_week:.0f}-{cumulative_weeks:.0f}")
                    except:
                        st.markdown(f"**{phase['name']}:** {duration}")
                else:
                    st.markdown(f"**{phase['name']}:** {duration}")
            
            st.info(f"Estimated total project duration: {cumulative_weeks:.0f} weeks")
        
        with col2:
            st.subheader("Phase Details")
            
            # Create tabs for each phase
            phase_tabs = st.tabs([phase["name"] for phase in roadmap])
            
            for i, tab in enumerate(phase_tabs):
                with tab:
                    phase = roadmap[i]
                    st.markdown(f"**Duration:** {phase['duration']}")
                    st.markdown("**Key Activities:**")
                    for activity in phase["activities"]:
                        st.markdown(f"- {activity}")
                    
                    # Add phase-specific notes
                    if phase["name"] == "Planning & Requirements":
                        st.info("Focus on capturing detailed requirements and establishing a clear project scope.")
                    elif phase["name"] == "Design":
                        st.info(f"Create detailed architecture diagrams for your {selected_arch_name} implementation.")
                    elif phase["name"] == "Development":
                        st.info(f"Set up a CI/CD pipeline early to streamline the development process.")
                    elif phase["name"] == "Testing":
                        st.info("Include automated testing to catch issues early.")
                    elif phase["name"] == "Deployment":
                        st.info("Implement monitoring and alerting before going live.")
                    elif phase["name"] == "Maintenance":
                        st.info("Plan for regular updates and security patches.")
    
    with tab5:
        st.header("Code Generator")
        
        st.markdown(f"""
        Generate code snippets for your {selected_arch_name} project using {selected_language}.
        """)
        
        component = st.radio("Component to generate", ["frontend", "backend", "database", "all"])
        
        if st.button("Generate Code"):
            with st.spinner("Generating code using Groq's LLM..."):
                code = generate_code_with_groq(
                    project_description, 
                    selected_arch_name,
                    selected_frontend,
                    selected_backend,
                    selected_database,
                    selected_language,
                    component
                )
                
                if code.startswith("Error"):
                    st.error(code)
                else:
                    st.code(code, language=selected_language.lower())
                    
                    if "```" in code:
                        # If the LLM returned code with markdown formatting, extract the actual code
                        clean_code = "\n".join([line for line in code.split("\n") 
                                              if not line.strip().startswith("```")])
                        
                        # Create a download button for the code
                        st.download_button(
                            label="Download Code",
                            data=clean_code,
                            file_name=f"{component}_code.{'py' if selected_language == 'Python' else 'js'}",
                            mime="text/plain"
                        )
                    else:
                        # Create a download button for the code
                        st.download_button(
                            label="Download Code",
                            data=code,
                            file_name=f"{component}_code.{'py' if selected_language == 'Python' else 'js'}",
                            mime="text/plain"
                        )
        
        st.info("The code generation feature uses Groq's LLM API to generate code based on your project description and selected technologies.")
        st.warning("Generated code is a starting point and may need customization for your specific requirements.")

def show_architecture_visualizer():
    st.title("Code Architecture Visualizer")
    st.write("Analyze and visualize your code architecture")
    
    # Code input
    code = st.text_area("Code Input", height=300, placeholder="Paste your code here to analyze its architecture...")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Auto-detect language and allow manual selection
        if code:
            detected_language = detect_language(code)
        else:
            detected_language = "Unknown"
            
        language = st.selectbox(
            "Programming Language",
            ["Auto-detect", "Python", "JavaScript", "TypeScript", "Java", "C#", "PHP", "Ruby", "Go"],
            index=0,
            help="Select the programming language or let the system detect it"
        )
        
        if language == "Auto-detect":
            if detected_language != "Unknown":
                st.info(f"Detected language: {detected_language}")
                language = detected_language
            else:
                st.warning("Language could not be auto-detected. Please select manually.")
                language = "Python"  # Default fallback
    
    with col2:
        parse_method = st.radio(
            "Parsing Method",
            ["Native Parser", "LLM Analysis (Groq)"],
            help="Choose how to analyze the code relationships"
        )
    
    with col3:
        diagram_type = st.radio(
            "Diagram Type",
            ["NetworkX", "Mermaid"],
            help="Choose the diagram visualization type"
        )
    
    # Show advanced options in an expander
    with st.expander("Advanced Options"):
        include_imports = st.checkbox("Include Imports in Diagram", value=True)
        include_methods = st.checkbox("Include Methods in Diagram", value=True)
        max_nodes = st.slider("Maximum Node Count", min_value=10, max_value=100, value=50)
        relation_threshold = st.slider("Relationship Detection Sensitivity", min_value=0.0, max_value=1.0, value=0.5, 
                                     help="Higher values filter out weaker relationships. Only applies to LLM analysis.")
    
    if st.button("Generate Diagram"):
        if not code:
            st.error("Please paste some code to analyze.")
        else:
            with st.spinner(f"Analyzing {language} code and generating diagram..."):
                try:
                    # Extract code components based on language and chosen method
                    use_llm = parse_method == "LLM Analysis (Groq)"
                    components = extract_components_by_language(code, language, use_llm)
                    
                    # Get additional analysis if LLM option selected
                    if use_llm:
                        llm_analysis = analyze_code_with_llm(code, language)
                        # Merge basic components with LLM analysis
                        analysis = {**components, **llm_analysis}
                    else:
                        # Create relationships based on basic parsing
                        analysis = components
                        analysis['relationships'] = []
                        
                        # Add simple relationship detection for non-LLM parsing
                        if 'classes' in components:
                            for cls in components['classes']:
                                # Generate dependency relationships based on attributes that might reference other classes
                                for attr in cls.get('attributes', []):
                                    for other_cls in components['classes']:
                                        if other_cls['name'] != cls['name'] and other_cls['name'].lower() in attr.lower():
                                            analysis['relationships'].append({
                                                'from': cls['name'],
                                                'to': other_cls['name'],
                                                'type': 'dependency'
                                            })
                    
                    # Apply filters based on advanced options
                    if not include_imports and 'imports' in analysis:
                        analysis['imports'] = []
                    
                    if not include_methods:
                        for cls in analysis.get('classes', []):
                            cls['methods'] = []
                    
                    # Limit nodes if needed
                    if 'classes' in analysis and len(analysis['classes']) > max_nodes:
                        analysis['classes'] = analysis['classes'][:max_nodes]
                    if 'functions' in analysis and len(analysis['functions']) > max_nodes:
                        analysis['functions'] = analysis['functions'][:max_nodes]
                    
                    # Generate the selected diagram type
                    if diagram_type == "Mermaid":
                        mermaid_code = generate_mermaid_diagram(analysis, language)
                        st.subheader(f"Mermaid Diagram for {language} Code")
                        st.code(mermaid_code, language="mermaid")
                        
                    else:  # NetworkX
                        fig = generate_networkx_diagram(analysis, language)
                        st.subheader(f"NetworkX Diagram for {language} Code")
                        st.pyplot(fig)
                        
                        # Add download button for the NetworkX diagram
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        btn = st.download_button(
                            label="Download Diagram as PNG",
                            data=buf,
                            file_name=f"code_architecture_{language}.png",
                            mime="image/png"
                        )
                    
                    # Display code components
                    with st.expander(f"{language} Code Components"):
                        tab1, tab2, tab3 = st.tabs(["Classes", "Functions", "Imports"])
                        
                        with tab1:
                            st.subheader("Classes")
                            if components.get('classes'):
                                for cls in components['classes']:
                                    st.write(f"**{cls['name']}**")
                                    if cls.get('parents'):
                                        st.write(f"Inherits from: {', '.join(cls['parents'])}")
                                    if cls.get('methods'):
                                        st.write(f"Methods: {', '.join(cls['methods'])}")
                                    if cls.get('attributes'):
                                        st.write(f"Attributes: {', '.join(cls['attributes'])}")
                                    st.write("---")
                            else:
                                st.write("No classes found.")
                        
                        with tab2:
                            st.subheader("Functions")
                            st.write(", ".join(components.get('functions', [])) if components.get('functions') else "None found")
                        
                        with tab3:
                            st.subheader("Imports")
                            st.write(", ".join(components.get('imports', [])) if components.get('imports') else "None found")
                    
                    # Display relationships
                    if 'relationships' in analysis and analysis['relationships']:
                        with st.expander("Component Relationships"):
                            for rel in analysis['relationships']:
                                if isinstance(rel, dict) and 'from' in rel and 'to' in rel:
                                    st.write(f"**{rel['from']}** → **{rel['to']}** ({rel.get('type', 'related')})")
                    
                    # Display raw LLM analysis if used
                    if parse_method == "LLM Analysis (Groq)" and "raw_analysis" in analysis:
                        with st.expander("Raw LLM Analysis"):
                            st.write(analysis["raw_analysis"])
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your code and try again.")

    # Add a footer with helpful information
    st.markdown("---")
    st.info("""
    **How to use this tool:**
    1. Paste your code in the text area
    2. Select or let the system auto-detect the programming language
    3. Choose between native parsing or LLM-powered analysis
    4. Select your preferred diagram type
    5. Adjust advanced options if needed
    6. Click "Generate Diagram" to visualize the code architecture
    """)
    
    # Adding sidebar content from paste-2.txt
    with st.sidebar:
        st.subheader("Supported Languages")
        st.markdown("""
        * **Python**: Full support with AST parsing
        * **JavaScript/TypeScript**: Support via regex parsing
        * **Java**: Support via regex parsing
        * **C#**: Support via regex parsing
        * **Other Languages**: Support via LLM analysis
        """)
        
        st.subheader("Tips for Better Results")
        st.markdown("""
        * Make sure your code is syntactically correct
        * For complex projects, analyze one file at a time
        * LLM analysis works better with comments in your code
        * Use "LLM Analysis" for more complete relationship detection
        * Adjust the relationship sensitivity for different levels of detail
        """)
        
        st.subheader("About")
        st.markdown("""
        This tool uses:
        - Language-specific parsers for code analysis
        - Groq LLM API for advanced code analysis
        - NetworkX and Matplotlib for diagram generation
        - Mermaid for alternative diagram format
        
        For large codebases, consider splitting the analysis into multiple files.
        """)

    # Add examples from paste-2.txt
    with st.expander("Example Code Snippets"):
        language_tabs = st.tabs(["Python", "JavaScript", "Java", "C#"])
        
        with language_tabs[0]:
            st.code("""
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
    
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color
    
    def speak(self):
        return "Meow!"

def create_animal(animal_type, name, **kwargs):
    if animal_type == "dog":
        return Dog(name, kwargs.get("breed", "unknown"))
    elif animal_type == "cat":
        return Cat(name, kwargs.get("color", "unknown"))
    else:
        return Animal(name)
            """, language="python")
        
        with language_tabs[1]:
            st.code("""
class Vehicle {
    constructor(make, model) {
        this.make = make;
        this.model = model;
    }
    
    drive() {
        console.log(`Driving ${this.make} ${this.model}`);
    }
}

class Car extends Vehicle {
    constructor(make, model, doors) {
        super(make, model);
        this.doors = doors;
    }
    
    honk() {
        console.log("Beep!");
    }
}

class Motorcycle extends Vehicle {
    constructor(make, model, type) {
        super(make, model);
        this.type = type;
    }
    
    wheelie() {
        console.log("Doing a wheelie!");
    }
}

function createVehicle(type, make, model, options) {
    if (type === "car") {
        return new Car(make, model, options.doors);
    } else if (type === "motorcycle") {
        return new Motorcycle(make, model, options.type);
    } else {
        return new Vehicle(make, model);
    }
}
            """, language="javascript")
        
        with language_tabs[2]:
            st.code("""
import java.util.ArrayList;
import java.util.List;

abstract class Shape {
    protected String color;
    
    public Shape(String color) {
        this.color = color;
    }
    
    public abstract double calculateArea();
    
    public String getColor() {
        return color;
    }
}

class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }
    
    public double getRadius() {
        return radius;
    }
}

class Rectangle extends Shape {
    private double width;
    private double height;
    
    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }
    
    @Override
    public double calculateArea() {
        return width * height;
    }
    
    public double getWidth() {
        return width;
    }
    
    public double getHeight() {
        return height;
    }
}

class ShapeFactory {
    public static Shape createShape(String type, String color, double... params) {
        if (type.equals("circle")) {
            return new Circle(color, params[0]);
        } else if (type.equals("rectangle")) {
            return new Rectangle(color, params[0], params[1]);
        }
        return null;
    }
}
            """, language="java")
            
        with language_tabs[3]:
            st.code("""
using System;
using System.Collections.Generic;

namespace ShapeExample
{
    public abstract class Shape
    {
        public string Color { get; protected set; }
        
        public Shape(string color)
        {
            Color = color;
        }
        
        public abstract double CalculateArea();
    }
    
    public class Circle : Shape
    {
        public double Radius { get; private set; }
        
        public Circle(string color, double radius) : base(color)
        {
            Radius = radius;
        }
        
        public override double CalculateArea()
        {
            return Math.PI * Radius * Radius;
        }
    }
    
    public class Rectangle : Shape
    {
        public double Width { get; private set; }
        public double Height { get; private set; }
        
        public Rectangle(string color, double width, double height) : base(color)
        {
            Width = width;
            Height = height;
        }
        
        public override double CalculateArea()
        {
            return Width * Height;
        }
    }
    
    public static class ShapeFactory
    {
        public static Shape CreateShape(string type, string color, params double[] dimensions)
        {
            switch (type.ToLower())
            {
                case "circle":
                    return new Circle(color, dimensions[0]);
                case "rectangle":
                    return new Rectangle(color, dimensions[0], dimensions[1]);
                default:
                    throw new ArgumentException("Invalid shape type");
            }
        }
    }
}
            """, language="csharp")
    # Add helpful information
 
st.markdown("""
    <style>
        /* MAIN PAGE BACKGROUND + TEXT */
        .stApp {
            background-color: #0b1e3f;
            color: white !important;
        }

        .stApp div, .stApp label, .stApp span, .stApp p, .stApp h1, .stApp h2, .stApp h3 {
            color: white !important;
        }

        /* INPUT FIELDS + TEXTAREAS + SELECT */
        input[type="text"], input[type="email"], input[type="password"], textarea {
            background-color: rgb(113, 117, 117) !important;
            color: white !important;
            border: 1px solid #ccc !important;
            border-radius: 5px !important;
            padding: 8px !important;
        }

        /* STREAMLIT SELECTBOX WRAPPER */
        div[data-baseweb="select"] {
            background-color: rgb(113, 117, 117) !important;
            border-radius: 5px !important;
        }

        /* TEXT inside selectbox */
        div[data-baseweb="select"] * {
            color: white !important;
        }

        /* DROPDOWN MENU OPTIONS */
        div[data-baseweb="menu"] {
            background-color: rgb(60, 63, 65) !important;
            color: white !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #d6f0ff !important;
        }

        section[data-testid="stSidebar"] * {
            color: black !important;
        }

        /* SIDEBAR INPUT FIELDS */
        section[data-testid="stSidebar"] input {
            background-color: rgb(113, 117, 117) !important;
            color: white !important;
        }

        /* SIDEBAR SELECTBOX STYLE */
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            background-color: rgb(113, 117, 117) !important;
            border-radius: 5px !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: white !important;
        }

        /* BUTTONS */
        .stButton>button {
            background-color: #457b9d !important;
            color: white !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] .stButton>button {
            background-color: #3099d2 !important;
            color: white !important;
        }

        /* HEADER */
        header[data-testid="stHeader"] {
            background-color: #0b1e3f;
        }

        /* DESCRIPTION BOXES */
        .description-box {
            background-color: #f0f0f0 !important;
            color: black !important;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 16px;
        }

        .description-box * {
            color: black !important;
        }

        .description-title {
            font-weight: bold;
            margin-bottom: 5px;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()