<script lang="ts">
	import InputBlock from "../modules/InputBlock.svelte";
	import ThemeToggle from "../components/ThemeToggle.svelte";
	import { global_state, input } from '../state.svelte';
	import { inputCallback } from "../callbacks.svelte";
</script>

<nav class="w-full fixed left-0 top-0 p-1 px-10 flex flex-row justify-between items-center z-50 bg-white">
	<a href="/" title="DoLTLMViz" class="font-mono text-2xl font-bold text-theme"> DoTLMViz </a>
	<InputBlock bind:value={input.text} inpEventCb={inputCallback} />
	<ThemeToggle bind:state={global_state.viewMode} style="z-[100]" leftlabel="Detailed" rightlabel="Overview"/>
</nav>
