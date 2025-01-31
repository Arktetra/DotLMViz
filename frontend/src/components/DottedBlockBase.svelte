<script lang="ts">
	import { scale } from 'svelte/transition';

	const {
		label = 'Untitled',
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();
</script>

<div
	in:scale={{ duration: 200 }}
	style="border-width: {borderSize};"
	class="relative m-2 h-full rounded-xl border-dashed border-theme transition-colors duration-200"
>
	<span
		class="absolute left-[50%] top-[-1.8rem] translate-x-[-50%] font-main text-lg uppercase text-gray-500 {titStyle}"
		>{label}</span
	>

	{#if children}
		<div in:scale={{ duration: 200 }} class="flex items-center {inStyle}">
			{@render children()}
		</div>
	{/if}
</div>
