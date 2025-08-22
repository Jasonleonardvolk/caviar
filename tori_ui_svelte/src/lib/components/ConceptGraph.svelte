<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  type GraphNode = d3.SimulationNodeDatum & {
    id: string;
    label: string;
    score?: number;
    method?: string;
    relationships_count?: number;
    type?: string;
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
  };

  type GraphLink = d3.SimulationLinkDatum<GraphNode> & {
    source: GraphNode | string;
    target: GraphNode | string;
    type?: string;
  };

  let svgEl: SVGSVGElement;
  let graphData: { nodes: GraphNode[]; edges: GraphLink[] } = { nodes: [], edges: [] };

  export { graphData };

  onMount(() => {
    if (!svgEl) return;

    const svg = d3.select<SVGSVGElement, unknown>(svgEl);
    
    // Clear any existing content
    svg.selectAll("*").remove();
    
    // Setup zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        root.attr('transform', event.transform.toString());
      });
    
    svg.call(zoom as any);

    // Create root group
    const root = svg.append('g').attr('class', 'graph-root');

    // Create link elements
    const link = root.append('g')
      .attr('class', 'links')
      .selectAll<SVGLineElement, GraphLink>('line')
      .data(graphData.edges)
      .enter().append('line')
      .attr('stroke', '#aaa')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d) => d.type === 'subject_of' ? 3 : 2);

    // Create node groups
    const node = root.append('g')
      .attr('class', 'nodes')
      .selectAll<SVGGElement, GraphNode>('g')
      .data(graphData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );

    // Add circles to nodes
    node.append('circle')
      .attr('r', (d) => Math.sqrt(d.relationships_count || 1) * 5 + 10)
      .attr('fill', (d) => {
        if (d.method?.includes('yake')) return '#4CAF50';
        if (d.method?.includes('spacy')) return '#2196F3';
        if (d.method?.includes('svo')) return '#FF9800';
        return '#9C27B0';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Add labels to nodes
    node.append('text')
      .text((d) => d.label)
      .attr('x', 0)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#333')
      .style('font-size', '12px')
      .style('font-family', 'sans-serif');

    // Add link labels
    const linkLabel = root.append('g')
      .attr('class', 'link-labels')
      .selectAll<SVGTextElement, GraphLink>('text')
      .data(graphData.edges)
      .enter().append('text')
      .text((d) => d.type || '')
      .attr('fill', '#666')
      .style('font-size', '10px');

    // Create simulation
    const simulation = d3.forceSimulation<GraphNode>(graphData.nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(graphData.edges)
        .id((d) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(400, 300))
      .force('collision', d3.forceCollide().radius(30))
      .on('tick', () => {
        // Update link positions
        link
          .attr('x1', (d) => (d.source as GraphNode).x || 0)
          .attr('y1', (d) => (d.source as GraphNode).y || 0)
          .attr('x2', (d) => (d.target as GraphNode).x || 0)
          .attr('y2', (d) => (d.target as GraphNode).y || 0);

        // Update link label positions
        linkLabel
          .attr('x', (d) => {
            const source = d.source as GraphNode;
            const target = d.target as GraphNode;
            return ((source.x || 0) + (target.x || 0)) / 2;
          })
          .attr('y', (d) => {
            const source = d.source as GraphNode;
            const target = d.target as GraphNode;
            return ((source.y || 0) + (target.y || 0)) / 2;
          });

        // Update node positions
        node.attr('transform', (d) => `translate(${d.x || 0},${d.y || 0})`);
      });

    // Add window resize handler
    const handleResize = () => {
      const width = svgEl.clientWidth;
      const height = svgEl.clientHeight;
      simulation.force('center', d3.forceCenter(width / 2, height / 2));
      simulation.alpha(0.3).restart();
    };

    window.addEventListener('resize', handleResize);

    // Cleanup on destroy
    return () => {
      window.removeEventListener('resize', handleResize);
      simulation.stop();
    };
  });
</script>

<style>
  svg {
    width: 100%;
    height: 100%;
    min-height: 600px;
    border: 1px solid #ddd;
    border-radius: 4px;
  }
  
  :global(.node) {
    cursor: move;
  }
  
  :global(.node circle) {
    transition: fill 0.3s;
  }
  
  :global(.node:hover circle) {
    fill-opacity: 0.8;
  }
</style>

<svg bind:this={svgEl} />
