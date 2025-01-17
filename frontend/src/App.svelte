<script lang="ts">
  import { onMount } from "svelte";

  import OutputBlock from "./modules/OutputBlock.svelte";
  import TransformerBlocks from "./modules/TransformerBlocks.svelte";
  import UnembeddingBlock from "./modules/UnembeddingBlock.svelte";
  import EmbeddingBlock from "./modules/EmbeddingBlock.svelte";
  import InputBlock from "./modules/InputBlock.svelte";
  import TokensBlock from "./modules/TokensBlock.svelte";

  import Navbar from "./lib/Navbar.svelte";
  import DottedBlockBase from "./components/DottedBlockBase.svelte";

  function getRand() {
    fetch("/api/rand")
      .then((d) => {  d.text()  })
      .then((d) => {  d })
      .catch((err) => { err;  });
  }

  onMount(getRand)
</script>

<Navbar />
<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <div class="flex flex-row justify-evenly items-center min-w-[90vw]">
    <TokensBlock />
    <DottedBlockBase label="GPT-2 Small" style="min-w-[50vw] flex flex-row justify-evenly items-center bg-none">
      <EmbeddingBlock />
      <TransformerBlocks />
      <UnembeddingBlock />
    </DottedBlockBase>
    <div class="flex flex-col justify-evenly items-center">
      <span class="text-2xl text-theme font-bold my-2 block">Prediction :</span>
      <span class="bg-theme rounded-md p-1 px-2 text-theme-w font-light">E</span>
    </div>
  </div>

  <OutputBlock />

  <InputBlock />
</section>