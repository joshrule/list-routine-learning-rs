var width = window.innerWidth,
    diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x),
    nodeSize = 17,
    root = d3.hierarchy(treeify_data(data, data.nodes[data.root], {move: "Empty", pruning: "None"})),
    best = -Infinity;

// get the number of visited nodes at each depth
// _.countBy(root.descendants().filter(n => n.data.type == "visited").map(n => n.depth), x => x)

let uid = 0;
root.descendants().reverse().forEach(d => {
    // give each node a unique id
    d.uid = uid++;
    // sort children by number of visits
    if (d.hasOwnProperty("children") && d.children !== null) {
        d.children.sort((x, y) => {
            if (!(x.data.hasOwnProperty("stats") || y.data.hasOwnProperty("stats"))){
                return 0;
            } else if (!x.data.hasOwnProperty("stats")) {
                return 1;
            } else if (!y.data.hasOwnProperty("stats")) {
                return -1;
            } else {
                return y.data.stats.q - x.data.stats.q;
            }
        });
    }
    // Toggle children off.
    d._children = d.children;
    if (d.depth) d.children = null;
    // Collect the best child.
    if (d.data.score !== null && d.data.score > best) best = d.data.score;
    // Correct scores mangled by JSON.
    if (d.data.score === null && d.data.state.playout !== "untried") {
        d.data.score = -Infinity;
    }
    if (d.data.hasOwnProperty("state") && d.data.state.playout !== "untried" && d.data.stats.q === null) {
        d.data.stats.q = -Infinity;
    }
});
root.x0 = 0;
root.x = 0;
root.y0 = 0;
root.y = 0;

const nodes = root.descendants();

let svg = d3.select("#svg-box")
    .append("svg")
    .attr("width", width)
    .attr("viewBox", [-20, -20, width, (nodes.length + 1) * nodeSize])
    .attr("font-family", "sans-serif")
    .attr("font-size", 10);

const gLink = svg.append("g")
      .attr("fill", "none")
      .attr("stroke", "#999");

const gNode = svg.append("g").attr("pointer-events", "all");

update(root);


function ser_node(node) {
    let node_string;
    switch (node.data.type) {
      case "unvisited": node_string = ser_unvisited(node); break;
      case "visited": node_string = ser_visited(node); break;
      case "failed": node_string = ser_failed(node); break;
      default: node_string = "unknown"; break;
    }
    let spacer = node_string.length > 0 ? "\t" : "";
    let move_string = node.data.move.move;
    return `${move_string}${spacer}${node_string}`;
}

function ser_unvisited() {
    return "";
}

function ser_failed() {
    return "";
}

function logsumexp(scores) {
    let corrected_scores = scores.map(s => s === null? -Infinity : s);
    let largest = d3.max(corrected_scores);
    let scaled_scores = corrected_scores.map(s => Math.exp(s - largest));
    return Math.log(d3.sum(scaled_scores)) + largest;
}

function ser_visited(node) {
    let data = node.data;
    let f = d3.format(".4f");
    let score = data.score === null ? data.score : f(data.score);
    let q = data.stats.q === null ? -Infinity : f(data.stats.q);
    let n = data.stats.n;
    let shared =
        `${data.handle}\t` +
        `${score}\t${q}\t${n}`;
    if (data.state.type === "revision") {
        return `${shared}\t${ser_revision(data.state)}`;
    } else if (data.state.type === "terminal") {
        return `${shared}\t${ser_terminal(data.state)}`;
    } else {
        return "unknown";
    }
}

function ser_revision(state) {
    var trs_string = state.playout.replace('/\n/g', ' ');
    return `\"${trs_string}\"`;
}

function ser_terminal(state) {
    var trs_string = state.trs.replace('\n', ' ');
    return `\"${trs_string}\"`;
}

function treeify_data(obj, node, move) {
    if (node === null) {
        let type = move.pruning === "Hard" ? "failed" : "unvisited";
        return {
            "type": type,
            "move": move,
            "children": [],
        };
    } else {
        let all_children = [...node.out];
        node.children = all_children.map(mh => {
            let new_move = obj.moves[mh];
            let ch = new_move.child;
            let new_node = ch === null ? ch : Object.assign({}, obj.nodes[ch]);
            return treeify_data(obj, new_node, new_move);
        });
        node.move = move;
        node.type = "visited";
        return node;
    }
}

function update(source) {
    const duration = d3.event && d3.event.altKey ? 2500 : 250;
    const nodes = root.descendants();
    const links = root.links();

    (() => {
        let i = 0;
        root.eachBefore(d => {
            d.index = i++;
            d.x = d.depth*nodeSize;
            d.y = d.index*nodeSize;
        });
    })();

    let height = nodes.length * nodeSize;

    var transition = d3.transition().duration(duration);

    // Update the nodes…
    let node = gNode.selectAll("g")
        .data(nodes, d => d.uid);

    // Enter any new nodes at the parent's previous position.
    let nodeEnter = node.enter().append("g")
        .attr("transform", d => `translate(${source.x0},${source.y0})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0);

    nodeEnter.append("circle")
        .attr("r", 2.5)
        .attr("fill", d => d._children ? "#1b9e77" : (d.data.type === "visited" ? "#333" : (d.data.type === "failed" ? "#d95f02" : "#ddd")))
        .attr("stroke-width", 10)
        .on("click", d => {
            d.children = d.children ? null : d._children;
            update(d);
        });

    nodeEnter.append("text")
        .attr("dy", "0.32em")
        .attr("x", d => 6)
        .html(ser_node);

    // Transition nodes to their new position.
    let nodeUpdate = node.merge(nodeEnter)
          .transition(transition)
          .attr("transform", d => `translate(${d.x},${d.y})`)
          .attr("fill-opacity", 1)
          .attr("stroke-opacity", 1);

    // Transition exiting nodes to the parent's new position.
    let nodeExit = node.exit().transition(transition).remove()
          .attr("transform", d => `translate(${source.x},${source.y})`)
          .attr("fill-opacity", 0)
          .attr("stroke-opacity", 0);

    // Update the links…
    let link = gLink.selectAll("path").data(links, d => d.target.uid);

    // Enter any new links at the parent's previous position.
    let linkEnter = link.enter().append("path")
        .attr("stroke-opacity", 0)
        .attr("stroke", "#555");

    // Transition links to their new position.
    link.merge(linkEnter).transition(transition)
        .attr("stroke-opacity", 1)
        .attr("d", d => `
            M${d.source.depth * nodeSize},${d.source.index * nodeSize}
            V${d.target.index * nodeSize}
            h${nodeSize}
          `);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition(transition).remove()
        .attr("stroke-opacity", 0);

    let max_width = d3.max(gNode.selectAll('text')._groups[0], function(n) {
        let box = n.getBoundingClientRect();
        return box.width + box.x;
    });

    svg.transition().duration(duration)
        .attr("viewBox", [-20, -20, max_width +20, height + 20])
        .attr("width", max_width + 40)
        .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));

    // Stash the old positions for transition.
    root.eachBefore(d => {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}

function expand(root) {
    if (root.hasOwnProperty("_children") && root.hasOwnProperty("children") && root.children == undefined) {
        root.children = root._children;
    }
    if (root.hasOwnProperty("children") && root.children !== undefined) {
        root.children.forEach(d => {
            d.children = d.children === null ? d._children : d.children;
            expand(d);
        });
    }
}

function collapse(root) {
    if (root.hasOwnProperty("children") && root.children !== undefined) {
        root.children = null;
    }
    if (root.hasOwnProperty("_children") && root._children !== undefined) {
        root._children.forEach(n => collapse(n));
    }
}
