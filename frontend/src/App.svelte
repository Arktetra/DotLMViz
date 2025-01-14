<script lang="ts">
  import { onMount } from "svelte";

  import OutputBlock from "./modules/OutputBlock.svelte";
  import TransformerBlocks from "./modules/TransformerBlocks.svelte";
  import UnembeddingBlock from "./modules/UnembeddingBlock.svelte";
  import EmbeddingBlock from "./modules/EmbeddingBlock.svelte";
  import InputBlock from "./modules/InputBlock.svelte";
  import TokensBlock from "./modules/TokensBlock.svelte";

  function getRand() {
    fetch("/api/rand")
      .then((d) => {  d.text()  })
      .then((d) => {  d })
      .catch((err) => { err;  });
  }

  onMount(getRand)
</script>

<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <div class="flex flex-row justify-evenly items-center min-w-[90vw]">
    <TokensBlock />
    <EmbeddingBlock />
    <TransformerBlocks />
    <UnembeddingBlock />
  </div>

  <OutputBlock />

  <InputBlock />
</section>