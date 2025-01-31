<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import Popup from '../components/Popup.svelte';
	import { global_state } from '../state.svelte';
	
	let { tokenInd = $bindable(), children = null } = $props();

	let popUpEnable: boolean = $state(false);

	const MAX_TOKEN_SIZE = 5;
	const MAX_TOKEN_COUNT = 20;

	const setPopUpState = (state: boolean) => {
		popUpEnable = state;
	};

	const tokenClick = (i: number) => {
		popUpEnable = true;
		tokenInd = i;
	};
</script>

<DottedBlockBase label="Tokens" inStyle="min-w-[6rem] min-h-[5rem] flex-col items-center">
	{#each global_state.tokens as token, ind}
		{#if ind < MAX_TOKEN_COUNT}
		<button
			onclick={() => tokenClick(ind)}
			class="my-1 block text-xl font-bold text-theme hover:underline hover:text-theme-alt"
		>
			{token.length > MAX_TOKEN_SIZE ? token.slice(0, MAX_TOKEN_SIZE) + '..' : token}
		</button>
		{/if}
	{/each}
	{#if global_state.tokens.length > MAX_TOKEN_COUNT}
		<span>...</span>
	{/if}
	{#if popUpEnable}
		<Popup onCloseCb={() => setPopUpState(false)}>
			{#if children}
				{@render children()}
			{/if}
		</Popup>
	{/if}
</DottedBlockBase>
