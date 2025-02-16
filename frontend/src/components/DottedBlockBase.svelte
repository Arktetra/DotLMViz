<script lang="ts">
	import { scale } from 'svelte/transition';

	const {
		label = '',
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();
</script>

<div
	in:scale={{ duration: 300 }}
	style="border-width: {borderSize};"
	class="relative h-full rounded-lg border-dashed border-theme transition-all duration-200 flex justify-center items-center {inStyle}"
>
	<span class="absolute left-[50%] top-[-1.5rem] translate-x-[-50%] uppercase font-light text-sm tracking-tighter {titStyle}">
		{label}
	</span>

	{#if children}
		{@render children()}
	{/if}
</div>
