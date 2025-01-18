<script lang="ts">
    import SideDrawer from "../components/SideDrawer.svelte"
    import ThemeInputSlider from "../components/ThemeInputSlider.svelte"

</script>

<SideDrawer  width={"25rem"} >
  <div class="w-full flex flex-col justify-evenly items-center h-full">
    <div class="w-full p-2 bg-theme-g rounded-md shadow-inner shadow-theme-g-alt">
      <span class="text-theme text-center block text-md font-extrabold underline mb-2 uppercase">Control Parameters</span>
      <ThemeInputSlider label={"Temperature"} min={-2} max={2} step={0.1}/>
      <hr class="my-1 border border-theme-w" />
      <ThemeInputSlider label={"Top K"} min={1} max={10} step={1} />
      <hr class="my-1 border border-theme-w" />
      <ThemeInputSlider label={"Top P"} min={0} max={1} step={0.05} />
    </div>
    <hr class="border border-theme w-full" />
    <h1 class="font-extrabold uppercase text-md my-2 text-center text-theme">Softmax Output</h1>
    <div class="w-full min-h-[15rem] p-3 flex flex-col justify-evenly items-center  bg-theme-g-alt rounded-md shadow-inner shadow-theme-g-alt">
      <div class="w-full text-ti font-light chart text-right">
          <span class="text-md underline block text-theme font-bold text-center my-4">Chart here.. Probability distribution</span>
      </div>
    </div>
    <span class="text-theme font-bold my-2">Next Token : <span class="bg-theme rounded-md p-1 px-2 text-theme-w font-light">E</span></span>
  </div>
</SideDrawer>

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