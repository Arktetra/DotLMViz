<script lang="ts">
	import InputBlock from "./InputBlock.svelte";
	import ThemeToggle from "../components/ThemeToggle.svelte";
	import { global_state, input } from '../state.svelte';
	import { inputCallback } from "../callbacks.svelte";
</script>

<nav class="w-full fixed left-0 top-0 px-5 flex flex-row justify-between items-center z-50 bg-white font-main-a">
	<a href="/" title="DoLTLMViz" class="text-xl font-extrabold text-theme">DoTLMViz </a>
	<InputBlock bind:value={input.text} inpEventCb={inputCallback} />
	<ThemeToggle bind:state={global_state.viewMode} style="z-50" leftlabel="Detailed" rightlabel="Overview"/>
</nav>
