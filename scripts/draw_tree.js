var width = window.innerWidth,
    height = 500,
    diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x),
    dx = 10,
    dy = width / 10,
    tree = d3.tree().nodeSize([dx, dy]),
    margin = ({top: 10, right: 120, bottom: 10, left: 40});

const root = d3.hierarchy(data);

root.x0 = dy / 2;
root.y0 = 0;
root.descendants().forEach((d, i) => {
    d.id = i;
    d._children = d.children;
    if (d.depth) d.children = null;
});

const svg = d3.select("body")
      .append("svg")
      .attr("viewBox", [-margin.left, -margin.top, width, dx])
      .attr("width", width)
      .attr("height", height)
      .style("font", "10px sans-serif")
      .style("user-select", "none");

const gLink = svg.append("g")
      .attr("fill", "none")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", 1.5);

const gNode = svg.append("g")
      .attr("cursor", "pointer")
      .attr("pointer-events", "all");

svg.call(d3.zoom().on("zoom", zoomed));

update(root);

function zoomed() {
    gLink.attr("transform", d3.event.transform);
    gNode.attr("transform", d3.event.transform);
}

function describe_node(d) {
    let div = d3.select("#info-box");
    var response = "";
    if (d.data.hasOwnProperty('score')) {
        response += "<li>" + "Score: " + d.data.score  + "</li>";
    }
    if (d.data.hasOwnProperty('q') && d.data.hasOwnProperty('n')) {
        response += "<li>" + "Q/N: " + +(Math.round(d.data.q + "e+3")  + "e-3") + "/" + d.data.n + "</li>";
    }
    if (d.data.hasOwnProperty('pruned')) {
        response += "<li>" + "Pruned: " + d.data.pruned  + "</li>";
    }
    if (d.data.hasOwnProperty('via')) {
        response += "<li>" + "Via: " + d.data.via + "</li>";
    }
    if (d.data.hasOwnProperty('trs')) {
        response += "<li>" + "TRS:<br/>" + d.data.trs.replace("/;/g", ";\n") + "</li>";
    }
    if (d.data.hasOwnProperty('playout')) {
        response += "<li>" + "Playout:<br/>" + d.data.playout.replace('/\n/g', '<br>') + "</li>";
    }
    console.log(response);
    if (response !== "") {
        div.html("<ul>" + response + "</ul>");
    }
}

function clear_description() {
    d3.select("#info-box").html("");
}

function update(source) {
    const duration = d3.event && d3.event.altKey ? 2500 : 250;
    const nodes = root.descendants().reverse();
    const links = root.links();

    // Compute the new tree layout.
    tree(root);

    let left = root;
    let right = root;
    root.eachBefore(node => {
        if (node.x < left.x) left = node;
        if (node.x > right.x) right = node;
    });

    const height = right.x - left.x + margin.top + margin.bottom;

    const transition = svg.transition()
          .duration(duration)
          .attr("viewBox", [-margin.left, left.x - margin.top, width, height])
          .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));

    // Update the nodes…
    const node = gNode.selectAll("g")
          .data(nodes, d => d.id);

    // Enter any new nodes at the parent's previous position.
    const nodeEnter = node.enter().append("g")
          .attr("transform", d => `translate(${source.y0},${source.x0})`)
          .attr("fill-opacity", 0)
          .attr("stroke-opacity", 0)
          .on("click", d => {
              d.children = d.children ? null : d._children;
              update(d);
          });

    nodeEnter.append("circle")
        .attr("r", 2.5)
        .attr("fill", d => d._children ? "#1b9e77" : (d.data.type === "visited" ? "#333" : "#ddd"))
        .attr("stroke-width", 10);

    // Transition nodes to their new position.
    const nodeUpdate = node.merge(nodeEnter)
          .on('mouseover', d => describe_node(d))
          .on('mouseout', () => clear_description())
          .transition(transition)
          .attr("transform", d => `translate(${d.y},${d.x})`)
          .attr("fill-opacity", 1)
          .attr("stroke-opacity", 1);

    // Transition exiting nodes to the parent's new position.
    const nodeExit = node.exit().transition(transition).remove()
          .attr("transform", d => `translate(${source.y},${source.x})`)
          .attr("fill-opacity", 0)
          .attr("stroke-opacity", 0);

    // Update the links…
    const link = gLink.selectAll("path")
          .data(links, d => d.target.id);

    // Enter any new links at the parent's previous position.
    const linkEnter = link.enter().append("path")
          .attr("stroke", d => d.target.data.type === "unvisited" ? "#ddd" : "#555")
          .attr("d", d => {
              const o = {x: source.x0, y: source.y0};
              return diagonal({source: o, target: o});
          });

    // Transition links to their new position.
    link.merge(linkEnter).transition(transition)
        .attr("d", diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition(transition).remove()
        .attr("d", d => {
            const o = {x: source.x, y: source.y};
            return diagonal({source: o, target: o});
        });

    // Stash the old positions for transition.
    root.eachBefore(d => {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}
