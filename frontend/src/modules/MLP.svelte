<script lang="ts">

    import BlockBase from "../components/BlockBase.svelte";
	import { getMLPPre } from "../routes/fetch.svelte";

	let _activeInd = $state(0);
</script>

<h1 class="w-full p-4 mb-5 text-2xl bg-theme-g text-theme font-bold text-center uppercase">Multi-level Perceptron (MLP)</h1>
<BlockBase label="Linear" href="/read/mlp-linear" clickEventCb={getMLPPre}/>
<BlockBase label="new_gelu" href="/read/mlp-new-gelu" />
<BlockBase label="Linear" href="/read/mlp-linear" />
<br />