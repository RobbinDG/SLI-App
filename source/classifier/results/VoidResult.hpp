#pragma once

namespace spp {

    /**
     * An abstract operation result object with a print menthod.
     */
    class VoidResult {
    public:
        /**
         * Prints the results of the operation.
         */
        virtual void print();
    };

}