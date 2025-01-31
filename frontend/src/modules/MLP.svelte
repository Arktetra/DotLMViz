<script lang="ts">
	import { ArrowRightAltOutline } from "flowbite-svelte-icons";
    import { MLPPreCallback } from "../callbacks.svelte";

    import BlockBase from "../components/BlockBase.svelte";
</script>

<h1 class="w-full p-4 mb-5 text-2xl bg-theme-g text-theme font-bold font-main-a  text-center uppercase">Multi-layer Perceptron (MLP)</h1>
<div class="flex flex-row h-[15rem] py-10 items-center">
    <ArrowRightAltOutline class="h-10 w-10 text-theme animate-pulse" />
    <BlockBase label="Linear" height={"75%"} width={"5rem"} href="/read/mlp-linear" clickEventCb={MLPPreCallback}/>
    <ArrowRightAltOutline class="h-6 w-6 text-theme animate-pulse" />
    <BlockBase label="GELU" height={"100%"} width={"7rem"} href="/read/mlp-gelu" />
    <ArrowRightAltOutline class="h-6 w-6 text-theme animate-pulse" />
    <BlockBase label="Linear" height={"75%"} width={"5rem"} href="/read/mlp-linear" />
    <ArrowRightAltOutline class="h-10 w-10 text-theme animate-pulse" />
</div>