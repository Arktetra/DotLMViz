<script lang="ts">
    import SideDrawer from "../components/SideDrawer.svelte";
    import ThemeInputSlider from "../components/ThemeInputSlider.svelte";
</script>

<SideDrawer width={"25rem"} >
    <div class="w-full flex flex-col justify-evenly items-center p-2 h-full">
      <div class="w-full bg-[#ccc] p-2 rounded-sm shadow-inner">
        <span class="text-theme text-center block text-lg font-bold underline">Control Parameters</span>
        <ThemeInputSlider label={"Temperature"} min={-2} max={2} step={0.1}/>
        <hr class="border border-theme-w" />
        <ThemeInputSlider label={"Top K"} min={1} max={10} step={1} />
        <hr class="border border-theme-w" />
        <ThemeInputSlider label={"Top P"} min={0} max={1} step={0.05} />
      </div>
      <hr class="border border-theme w-full" />
      <h1 class="font-bold uppercase text-xl my-2 text-center text-theme">Output</h1>
      <div class="w-full min-h-[10rem] flex flex-col justify-evenly items-center">
        <div class=" w-[100%] p-[5rem] py-[10rem] bg-[#999] rounded-md my-5 text-theme-w">
          bar chart here...
        </div>
        <span>Prediction : <span class="bg-theme rounded-md p-2 px-3 text-theme-w">test</span></span>
      </div>
    </div>
</SideDrawer>