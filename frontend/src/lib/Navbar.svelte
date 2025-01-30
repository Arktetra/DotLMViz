<script lang="ts">
	import InputBlock from "../modules/InputBlock.svelte";
	import ThemeToggle from "../components/ThemeToggle.svelte";
	import { getTokens } from '../routes/fetch.svelte';
	import { global_state, input } from '../state.svelte';

	const genToken = async () => {
		await getTokens(input.text);
	};

	const onInpChange = (v: string) => {
		input.text = v;
		genToken();
		input.isChanged = true;
	};

	$effect(() => {
		genToken();
	});

</script>

<nav class="w-full fixed left-0 top-0 p-1 px-10 flex flex-row justify-between items-center z-50 bg-white">
	<a href="/" title="DoLTLMViz" class="font-mono text-2xl font-bold text-theme"> DoLTLMViz </a>
	<InputBlock bind:value={input.text} inpEventCb={onInpChange} />
	<ThemeToggle bind:state={global_state.viewMode} style="z-[100]" leftlabel="Detailed" rightlabel="Overview"/>
</nav>
