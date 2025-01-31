<script lang="ts">
	import { QuestionCircleSolid, ExpandOutline, MinimizeOutline } from 'flowbite-svelte-icons';
	import { fade } from 'svelte/transition';

	const {
		label = 'Untitled',
        href = "/",
        expandCb = null,
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();

	const registerExpand = () => {
		if(expandCb) expandCb()
		expanded = !expanded;
	}

	let expanded: boolean = $state(false);
</script>

<div
	style="border-width: {borderSize};"
	class="relative m-2 rounded-xl border-dashed border-theme transition-colors duration-200"
>
	<div class="absolute end-1 top-[0.2rem] text-theme flex flex-row justify-evenly items-center">
		<a {href} target="_blank" title={label} on:click|stopPropagation class="mx-2">
			<QuestionCircleSolid size={'sm'} />
		</a>
		<button on:click|stopPropagation={registerExpand}>
			{#if expanded}
				<MinimizeOutline size={'lg'} />
			{:else}
				<ExpandOutline size={'lg'} />
			{/if}
		</button>
	</div>
	<span
		class="absolute left-[50%] top-[-1.8rem] translate-x-[-50%] uppercase text-gray-500 {titStyle}"
	>
		{label}
	</span>

	{#if children}
		<div transition:fade={{duration:500, delay:500}} class="flex items-center transition-all duration-500 pt-12 {inStyle}">
			{@render children()}
		</div>
	{/if}
</div>
