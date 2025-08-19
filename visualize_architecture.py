import graphviz

def create_mcp_tori_diagram():
    dot = graphviz.Digraph('MCP-TORI Architecture', comment='Integration Flow')
    dot.attr(rankdir='LR', bgcolor='lightgray')
    
    # Define node styles
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    # Frontend
    dot.node('UI', 'Svelte Frontend\n:5173', fillcolor='lightblue')
    
    # Python Backend
    dot.node('PY', 'Python Backend\nrun_stable_server.py\n:8002', fillcolor='lightgreen')
    dot.node('TORI', 'TORI Pipeline\npipeline.py', fillcolor='yellow')
    dot.node('BRIDGE', 'MCP Bridge\nmcp_bridge_real_tori.py', fillcolor='orange')
    
    # MCP Architecture  
    dot.node('MCP', 'MCP Gateway\n:8080', fillcolor='lightcoral')
    dot.node('TK', 'Trust Kernel', fillcolor='pink')
    dot.node('KAIZEN', 'MCP:kaizen', fillcolor='lavender')
    dot.node('CELERY', 'MCP:celery', fillcolor='lavender')
    
    # Connections
    dot.edge('UI', 'PY', label='User Input')
    dot.edge('PY', 'TORI', label='Filter')
    dot.edge('TORI', 'BRIDGE', label='Filtered')
    dot.edge('BRIDGE', 'MCP', label='HTTP/WS')
    dot.edge('MCP', 'TK', label='Verify')
    dot.edge('TK', 'KAIZEN')
    dot.edge('TK', 'CELERY')
    
    # Return path
    dot.edge('MCP', 'BRIDGE', label='Response', style='dashed')
    dot.edge('BRIDGE', 'TORI', label='Filter', style='dashed')
    dot.edge('TORI', 'PY', label='Safe', style='dashed')
    dot.edge('PY', 'UI', label='Display', style='dashed')
    
    # Render
    dot.render('mcp_tori_architecture', format='png', cleanup=True)
    print("✅ Architecture diagram saved as 'mcp_tori_architecture.png'")

if __name__ == "__main__":
    create_mcp_tori_diagram()