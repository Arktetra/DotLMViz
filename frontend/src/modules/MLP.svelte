<script lang="ts">
	import { MLPPreCallback } from "../callbacks.svelte";

    import BlockBase from "../components/BlockBase.svelte";
</script>

<h1 class="w-full p-4 mb-5 text-2xl bg-theme-g text-theme font-bold text-center uppercase">Multi-level Perceptron (MLP)</h1>
<BlockBase label="Linear" href="/read/mlp-linear" clickEventCb={MLPPreCallback}/>
<BlockBase label="new_gelu" href="/read/mlp-new-gelu" />
<BlockBase label="Linear" href="/read/mlp-linear" />
<br />