<script lang="ts">
    import SideDrawer from "../components/SideDrawer.svelte";
    import ThemeInputSlider from "../components/ThemeInputSlider.svelte";

    import { onMount } from 'svelte';
    import * as d3 from 'd3'
    import Popup from "../components/Popup.svelte";
    
    const alphabet = [{"token":"A","prob":0.08167},{"token":"B","prob":0.01492},{"token":"C","prob":0.02782},{"token":"D","prob":0.04253},{"token":"E","prob":0.12702},{"token":"F","prob":0.02288},{"token":"G","prob":0.02015},{"token":"H","prob":0.06094},{"token":"I","prob":0.06966},{"token":"J","prob":0.00153},{"token":"K","prob":0.00772},{"token":"L","prob":0.04025},{"token":"M","prob":0.02406},{"token":"N","prob":0.06749},{"token":"O","prob":0.07507},{"token":"P","prob":0.01929},{"token":"Q","prob":0.00095},{"token":"R","prob":0.05987},{"token":"S","prob":0.06327},{"token":"T","prob":0.09056},{"token":"U","prob":0.02758},{"token":"V","prob":0.00978},{"token":"W","prob":0.0236},{"token":"X","prob":0.0015},{"token":"Y","prob":0.01974},{"token":"Z","prob":0.00074}];

    let el : any;
    let elMax : any;
    let outputMax: boolean = $state(false);

    const chart = () => {
      // Specify the chart’s dimensions, based on a bar’s height.
      const barHeight = 25;
      const marginTop = 30;
      const marginRight = 0;
      const marginBottom = 10;
      const marginLeft = 30;
      const width = 928;
      const height = Math.ceil((alphabet.length + 0.1) * barHeight) + marginTop + marginBottom;

      // Create the scales.
      const x = d3.scaleLinear()
          .domain([0, d3.max(alphabet, d => d.prob)])
          .range([marginLeft, width - marginRight]);
      
      const y = d3.scaleBand()
          .domain(d3.sort(alphabet, d => -d.prob).map(d => d.token))
          .rangeRound([marginTop, height - marginBottom])
          .padding(0.1);

      // Create a value format.
      const format = x.tickFormat(20, "");

      // Create the SVG container.
      const svg = d3.create("svg")
          .attr("width", width)
          .attr("height", height)
          .attr("viewBox", [0, 0, width, height])
          .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;");
      
      // Append a rect for each letter.
      svg.append("g")
          .attr("fill", "steelblue")
        .selectAll()
        .data(alphabet)
        .join("rect")
          .attr("x", x(0))
          .attr("y", (d) => y(d.token))
          .attr("width", (d) => x(d.prob) - x(0))
          .attr("height", y.bandwidth());
      
      // Append a label for each letter.
      svg.append("g")
          .attr("fill", "white")
          .attr("text-anchor", "end")
        .selectAll()
        .data(alphabet)
        .join("text")
          .attr("x", (d) => x(d.prob))
          .attr("y", (d) => y(d.token) + y.bandwidth() / 2)
          .attr("dy", "0.35em")
          .attr("dx", -4)
          .text((d) => format(d.prob))
        .call((text) => text.filter(d => x(d.prob) - x(0) < 20) // short bars
          .attr("dx", +4)
          .attr("fill", "black")
          .attr("text-anchor", "start"));

      // Create the axes.
      svg.append("g")
          .attr("transform", `translate(0,${marginTop})`)
          .call(d3.axisTop(x).ticks(width / 80, "%"))
          .call(g => g.select(".domain").remove());

      svg.append("g")
          .attr("transform", `translate(${marginLeft},0)`)
          .call(d3.axisLeft(y).tickSizeOuter(0));

      return svg.node();
    } 

    onMount(()=> {
      d3.select(el).append(chart);
    })

    $effect(()=>{
      outputMax ? d3.select(elMax).append(chart) : ()=>{};
    })

</script>

<SideDrawer  width={"25rem"} >
  <div class="w-full flex flex-col justify-evenly items-center h-full">
    <div class="w-full p-2 bg-theme-g rounded-md shadow-inner shadow-theme-g-alt">
      <span class="text-theme text-center block text-md font-bold underline">Control Parameters</span>
      <ThemeInputSlider label={"Temperature"} min={-2} max={2} step={0.1}/>
      <hr class="my-1 border border-theme-w" />
      <ThemeInputSlider label={"Top K"} min={1} max={10} step={1} />
      <hr class="my-1 border border-theme-w" />
      <ThemeInputSlider label={"Top P"} min={0} max={1} step={0.05} />
    </div>
    <hr class="border border-theme w-full" />
    <h1 class="font-extrabold uppercase text-xl my-2 text-center text-theme">Output</h1>
    <div class="w-full min-h-[10rem] p-3 flex flex-col justify-evenly items-center  bg-theme-g rounded-md shadow-inner shadow-theme-g-alt">
      <div bind:this={el} class="w-full text-[0.70rem] font-light chart text-right" onclick={()=>outputMax=true}>
          <span class="text-md underline block text-theme font-bold text-center my-4">Distribution</span>
      </div>
    </div>
    <span class="text-theme font-bold my-2">Prediction : <span class="bg-theme rounded-md p-1 px-2 text-theme-w font-light">E</span></span>
  </div>
</SideDrawer>
  
{#if outputMax}
  <Popup onClose={()=>outputMax=false}>
    <div bind:this={elMax} class="w-full p-4 text-[0.70rem] font-light chart text-right text-theme-w">
      <span class="text-2xl underline block text-theme-w font-bold text-center my-4">Distribution</span>
    </div>
  </Popup>
{/if}


<style lang="css">
	.chart :global(div) {
		background-color: #665191;
		padding: 3px;
		margin: 1px;
		color: white;
    opacity: 50%;
	}

  .chart :global(div):hover {
    opacity: 100%;
  }
</style>