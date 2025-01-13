<script lang="ts">
  import { onMount } from "svelte";
  import ThemeButton from "./components/ThemeButton.svelte";
  import ThemeInputField from "./components/ThemeInputField.svelte";
  import SideDrawer from "./components/SideDrawer.svelte";
  import ThemeInputSlider from "./components/ThemeInputSlider.svelte";
  import BlockBase from "./components/BlockBase.svelte";
  import DottedBlockBase from "./components/DottedBlockBase.svelte";

  function getRand() {
    fetch("/api/rand")
      .then((d) => {  d.text()  })
      .then((d) => {  d })
      .catch((err) => { err;  });
  }

  onMount(getRand)

  const _embeddings = [
    {
      label: "Token Embedding",
      redirect: "/readings/tokenembedding"
    },
    {
      label: "Positional Embedding",
      redirect: "/readings/positionalembedding"
    }
  ]

  const _transformerBlock = [
    {
      label: "Attention Head",
      redirect: "/readings/attentionhead"
    },
    {
      label: "MLP",
      redirect: "/readings/mlp"
    }
  ]
</script>

<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <div class="flex flex-row justify-evenly items-center min-w-[90vw]">
    <DottedBlockBase label="Embeddings">
      {#each _embeddings as item}
        <BlockBase label={item.label} >
          <h1 class="bg-theme p-4 ">{item.label}</h1>
        </BlockBase>
      {/each}
    </DottedBlockBase>

    <DottedBlockBase label="Transformer Blocks" style="flex flex-row justify-between items-center w-[30rem] h-[20rem]">
      {#each _transformerBlock as item}
        <BlockBase label={item.label} >
          <h1 class="bg-theme p-4 ">{item.label}</h1>
        </BlockBase>
      {/each}
    </DottedBlockBase>
    
    <DottedBlockBase label="Unembedding">
      <BlockBase label="Unembedding" width={"20rem"} height={"20rem"} >
        <h1 class="bg-theme p-4 "></h1>
      </BlockBase>
    </DottedBlockBase>
  </div>

  <SideDrawer width={"25rem"} >
    <div class="w-full flex flex-col justify-evenly items-center p-2 h-full">
      <h1 class="font-bold uppercase text-xl my-2 text-center text-theme">Output</h1>
      <div class="w-full bg-[#ccc] p-2 rounded-sm shadow-inner">
        <span class="text-theme text-center block text-lg font-bold underline">Control Parameters</span>
        <ThemeInputSlider label={"Temperature"} min={-2} max={2} step={0.1}/>
        <hr class="border border-theme-w" />
        <ThemeInputSlider label={"Top K"} min={1} max={10} step={1} />
        <hr class="border border-theme-w" />
        <ThemeInputSlider label={"Top P"} min={0} max={1} step={0.05} />
      </div>
      <hr class="border border-theme w-full" />
      <div class="w-full min-h-[10rem] flex flex-col justify-evenly items-center">
        <h1 class="text-xl text-theme font-bold">Probability Distribution</h1>
        <div class=" w-[100%] p-[5rem] py-[10rem] bg-[#999] rounded-md my-5 text-theme-w">
          bar chart here...
        </div>
        <span>Prediction : <span class="bg-theme rounded-md p-2 px-3 text-theme-w">test</span></span>
      </div>
    </div>
  </SideDrawer>
  
  <div>
    <ThemeInputField />
    <ThemeButton label="Generate" />
  </div>
</section>